# Loss functions

from cv2 import mean
import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel
import math
import torch.nn.functional as F
from utils.general import xyxy2xywh, xywh2xyxy,xywhn2xyxy
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        # BCEkl = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['kl_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=0.0)

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            # BCEcls, BCEobj, BCEkl= FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCEkl, g)
            BCEcls, BCEobj= FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)


        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        # self.BCEcls, self.BCEobj, self.BCEkl, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj,BCEkl, model.gr, h, autobalance
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # lkl = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # tcls_teacher, tbox_teacher, indices_teacher, anchors_teacher = self.build_targets(p_teacher, targets)  # targets
        # for i in range(len(tcls)):
        #     if not len(tcls[i]==0):
        #         lkl += self.BCEkl(tbox[i], tbox_teacher[i])
            # kl_loss += torch.nn.MSELoss()(pred[i], pred_teacher[i])
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5 #取出每幅图像包含目标的网格和对应anchor的预测结果，对目标位置的预测值进行sigmoid运算后乘以2再减去0.5得到box中心点的位置
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i] #对目标宽高预测值进行sigmoid运算后乘以2再平方再乘以对应的anchor的宽高得到预测框的宽高值
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target) .T 矩阵转置
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, lbox , lobj , lcls #torch.cat((lbox, lobj, lcls, loss)).detach() # .detach_() 和 .data用于切断反向传播

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt) 
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain 获得当前输出层的宽高[128,128,128,128]

            # Match targets to anchors
            t = targets * gain # 真实框相对坐标*输出层大小 = 真实框的绝对坐标  [num_anchors, num_targets, image_index + cls_id + bbox + anchor_index]
            if nt:
                # Matches
                #这个部分是计算gt和anchor的匹配程度
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio # wh ratio  真实框wh/锚框wh  [num_anchors, num_targets, w+h]
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare 找出锚框与真实框高和宽的比值都小于4的锚框
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                #将满足条件的targets筛选出来。
                t = t[j]  # filter

                # Offsets
                # 找到这是框的中心点，再添加两个距离它最近的两个点也作为正样本
                #这个部分就是扩充targets的数量，将比较targets附近的4个点，选取最近的2个点作为新targets中心，新targets的w、h使用与原targets一致，只是中心点坐标的不同。
                # 参考博文https://blog.csdn.net/cdknight_happy/article/details/109817548#t11
                # x小于0.5就靠近左边的网格，y小于0.5就靠近上边的网格
                gxy = t[:, 2:4]  # grid xy 真实框中心坐标x, y
                gxi = gain[[2, 3]] - gxy  # inverse 求反，通过下面的判断求出中心点偏移的方向
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T # gxy % 1. 意思就是求得坐标xy后的小数点，也就是相对每个网格的偏移量  j代表x，k代表y
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T  ## gxi % 1. < g 偏移量不能超过0.5，
                j = torch.stack((torch.ones_like(j), j, k, l, m)) ## [5, num_targets] 包括target中心点和他的四个相邻网格（下、右、上、左）
                t = t.repeat((5, 1, 1))[j] ##筛选后t的数量是原来t的3倍。
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] ## 选择网格
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices  第几个网格为正样本

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices 该网格是哪张图片的，并由哪个锚框进行预测
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box 添加真实框的中心点坐标相对于所在网格的偏移量，宽高
            anch.append(anchors[a])  # anchors 添加锚框
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1).cuda())

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        # size = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project).reshape(-1, 4)
        return x

class ComputeLoss_dis:
    # Compute losses
    def __init__(self, model, autobalance=False,reg_max=16):
        super(ComputeLoss_dis, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        # BCEkl = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['kl_pw']], device=device))
        DFL = DistributionFocalLoss()

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=0.0)

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            # BCEcls, BCEobj, BCEkl= FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCEkl, g)
            BCEcls, BCEobj= FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)


        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        # self.BCEcls, self.BCEobj, self.BCEkl, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj,BCEkl, model.gr, h, autobalance
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        self.DFL = DFL
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))
        self.reg_max = reg_max
        self.integral = Integral(self.reg_max)

    def __call__(self, x, targets):  # predictions, targets, model
        p = []
        for i in range(self.nl): 
            p.append(torch.zeros([x[i].shape[0],self.nl,x[i].shape[2],x[i].shape[3],self.nc + 5]).to(x[i].device)) 
        for i in range(len(x)):
            p[i][:,:,:,:,:2] = self.integral(x[i][:,:,:,:,:2 * (self.reg_max + 1)]).reshape(x[i].shape[0],x[i].shape[1],x[i].shape[2],x[i].shape[3],2)
            p[i][:,:,:,:,2:] = x[i][:,:,:,:,2 * (self.reg_max + 1):]
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        ldis = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # tcls_teacher, tbox_teacher, indices_teacher, anchors_teacher = self.build_targets(p_teacher, targets)  # targets
        # for i in range(len(tcls)):
        #     if not len(tcls[i]==0):
        #         lkl += self.BCEkl(tbox[i], tbox_teacher[i])
            # kl_loss += torch.nn.MSELoss()(pred[i], pred_teacher[i])
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                xs = x[i][b, a, gj, gi]

                # Regression
                pxy = ps[:, :2] #.sigmoid() * 2. - 0.5 #取出每幅图像包含目标的网格和对应anchor的预测结果，对目标位置的预测值进行sigmoid运算后乘以2再减去0.5得到box中心点的位置
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i] #对目标宽高预测值进行sigmoid运算后乘以2再平方再乘以对应的anchor的宽高得到预测框的宽高值
                xxy = xs[:,:2 * (self.reg_max + 1)]
                # xwh = xs[:, 2 * (self.reg_max + 1):4 * (self.reg_max + 1)] * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                ldis += self.DFL(xxy.reshape(-1, self.reg_max + 1),tbox[i][:,:2].reshape(-1)).mean()
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target) .T 矩阵转置
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + ldis
        return loss * bs, lbox , lobj , lcls , ldis #torch.cat((lbox, lobj, lcls, loss)).detach() # .detach_() 和 .data用于切断反向传播

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt) 
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain 获得当前输出层的宽高[128,128,128,128]

            # Match targets to anchors
            t = targets * gain # 真实框相对坐标*输出层大小 = 真实框的绝对坐标  [num_anchors, num_targets, image_index + cls_id + bbox + anchor_index]
            if nt:
                # Matches
                #这个部分是计算gt和anchor的匹配程度
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio # wh ratio  真实框wh/锚框wh  [num_anchors, num_targets, w+h]
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare 找出锚框与真实框高和宽的比值都小于4的锚框
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                #将满足条件的targets筛选出来。
                t = t[j]  # filter

                # Offsets
                # 找到这是框的中心点，再添加两个距离它最近的两个点也作为正样本
                #这个部分就是扩充targets的数量，将比较targets附近的4个点，选取最近的2个点作为新targets中心，新targets的w、h使用与原targets一致，只是中心点坐标的不同。
                # 参考博文https://blog.csdn.net/cdknight_happy/article/details/109817548#t11
                # x小于0.5就靠近左边的网格，y小于0.5就靠近上边的网格
                gxy = t[:, 2:4]  # grid xy 真实框中心坐标x, y
                gxi = gain[[2, 3]] - gxy  # inverse 求反，通过下面的判断求出中心点偏移的方向
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T # gxy % 1. 意思就是求得坐标xy后的小数点，也就是相对每个网格的偏移量  j代表x，k代表y
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T  ## gxi % 1. < g 偏移量不能超过0.5，
                j = torch.stack((torch.ones_like(j), j, k, l, m)) ## [5, num_targets] 包括target中心点和他的四个相邻网格（下、右、上、左）
                t = t.repeat((5, 1, 1))[j] ##筛选后t的数量是原来t的3倍。
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] ## 选择网格
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices  第几个网格为正样本

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices 该网格是哪张图片的，并由哪个锚框进行预测
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box 添加真实框的中心点坐标相对于所在网格的偏移量，宽高
            anch.append(anchors[a])  # anchors 添加锚框
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

class LevelAttention_loss(nn.Module):

    def forward(self, img_batch_shape, attention_mask, target):

        h, w = img_batch_shape[2], img_batch_shape[3]

        #mask_losses = []
        mask_loss = 0

        batch_size = img_batch_shape[0]
        n = target.shape[0]  # number of targets
        if n:
            for j in range(batch_size):
                try:
                    n = min([i for i in range(target.shape[0]) if target[i,0]==j])
                    m = max([i for i in range(target.shape[0]) if target[i,0]==j])
                except:
                    continue
                
                bbox_annotation = xywhn2xyxy(target[n:m+1,2:], w=w, h=h, padw=0, padh=0) #bboxs[j, :, :]
                #bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

                cond1 = torch.le(bbox_annotation[:, 0], w)
                cond2 = torch.le(bbox_annotation[:, 1], h)
                cond3 = torch.le(bbox_annotation[:, 2], w)
                cond4 = torch.le(bbox_annotation[:, 3], h)
                cond = cond1 * cond2 * cond3 * cond4

                bbox_annotation = bbox_annotation[cond, :]

                # if bbox_annotation.shape[0] == 0:
                #     mask_losses.append(torch.tensor(0).float().cuda())
                #     continue

                #bbox_area = (bbox_annotation[:, 2] - bbox_annotation[:, 0]) * (bbox_annotation[:, 3] - bbox_annotation[:, 1])

                # mask_loss = []

                #for id in range(len(attention_mask)):
                ################
                #attention_map = attention_mask[id][j, 0, :, :]
                attention_map = attention_mask[j, 0, :, :]

                # min_area = (2 ** (id + 5)) ** 2 * 0.5
                # max_area = (2 ** (id + 5) * 1.58) ** 2 * 2

                # level_bbox_indice1 = torch.ge(bbox_area, min_area)
                # level_bbox_indice2 = torch.le(bbox_area, max_area)

                # level_bbox_indice = level_bbox_indice1 * level_bbox_indice2

                # level_bbox_annotation = bbox_annotation[level_bbox_indice, :].clone()

                level_bbox_annotation = bbox_annotation.clone()

                attention_h, attention_w = attention_map.shape

                if level_bbox_annotation.shape[0]:
                    level_bbox_annotation[:, 0] *= attention_w / w
                    level_bbox_annotation[:, 1] *= attention_h / h
                    level_bbox_annotation[:, 2] *= attention_w / w
                    level_bbox_annotation[:, 3] *= attention_h / h

                mask_gt = torch.zeros(attention_map.shape)
                mask_gt = mask_gt.cuda()

                for i in range(level_bbox_annotation.shape[0]):

                    x1 = max(int(level_bbox_annotation[i, 0]), 0)
                    y1 = max(int(level_bbox_annotation[i, 1]), 0)
                    x2 = min(math.ceil(level_bbox_annotation[i, 2]) + 1, attention_w)
                    y2 = min(math.ceil(level_bbox_annotation[i, 3]) + 1, attention_h)

                    mask_gt[y1:y2, x1:x2] = 1

                mask_gt = mask_gt[mask_gt >= 0]
                mask_predict = attention_map[attention_map >= 0]

                #mask_loss.append(F.binary_cross_entropy(mask_predict, mask_gt))
                #mask_loss.append(nn.BCEWithLogitsLoss()(mask_predict, mask_gt))
                mask_loss += nn.BCEWithLogitsLoss()(mask_predict, mask_gt)
                #################
                #mask_losses.append(torch.stack(mask_loss).mean())

        return mask_loss #torch.stack(mask_losses).mean(dim=0, keepdim=True)

class L1Loss(object):
    def __call__(self, input, target):
        return torch.abs(input - target).mean()

#KnowledgeDistillation loss
class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(
            pred,
            soft_label,
            #weight,
            #reduction=reduction,
            #avg_factor=avg_factor,
            T=self.T)

        return loss_kd

def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       T,
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (
            T * T)

    return kd_loss

# def reduce_loss(loss, reduction):
#     """Reduce loss as specified.

#     Args:
#         loss (Tensor): Elementwise loss tensor.
#         reduction (str): Options are "none", "mean" and "sum".

#     Return:
#         Tensor: Reduced loss tensor.
#     """
#     reduction_enum = F._Reduction.get_enum(reduction)
#     # none: 0, elementwise_mean:1, sum: 2
#     if reduction_enum == 0:
#         return loss
#     elif reduction_enum == 1:
#         return loss.mean()
#     elif reduction_enum == 2:
#         return loss.sum()

# def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
#     """Apply element-wise weight and reduce loss.

#     Args:
#         loss (Tensor): Element-wise loss.
#         weight (Tensor): Element-wise weights.
#         reduction (str): Same as built-in losses of PyTorch.
#         avg_factor (float): Avarage factor when computing the mean of losses.

#     Returns:
#         Tensor: Processed loss values.
#     """
#     # if weight is specified, apply element-wise weight
#     if weight is not None:
#         loss = loss * weight

#     # if avg_factor is not specified, just reduce the loss
#     if avg_factor is None:
#         loss = reduce_loss(loss, reduction)
#     else:
#         # if reduction is mean, then average the loss by avg_factor
#         if reduction == 'mean':
#             loss = loss.sum() / avg_factor
#         # if reduction is 'none', then do nothing, otherwise raise an error
#         elif reduction != 'none':
#             raise ValueError('avg_factor can not be used with reduction="sum"')
#     return loss

class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss (DFL) is a variant of
    Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes
    for Dense Object Detection
    https://arxiv.org/abs/2006.04388

    Args:
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target, 
            # weight, 
            # reduction=reduction, 
            # avg_factor=avg_factor
            )
        return loss_cls

def distribution_focal_loss(pred, label):
    """Distribution Focal Loss (DFL) is from
    Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes
    for Dense Object Detection
    https://arxiv.org/abs/2006.04388

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).

    Return:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
        + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss





class BDC(nn.Module):
    """modified from Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification.
    https://github.com/Fei-Long121/DeepBDC
    """
    def __init__(self, T=1): #, is_vec=True, input_dim=[640, 10, 10], dimension_reduction=None, activate='relu'
        super(BDC, self).__init__()
        self.T = T
        # self.is_vec = is_vec
        # self.dr = dimension_reduction
        # self.activate = activate
        # self.input_dim = input_dim[0]
        # if self.dr is not None and self.dr != self.input_dim:
        #     if activate == 'relu':
        #         self.act = nn.ReLU(inplace=True)
        #     elif activate == 'leaky_relu':
        #         self.act = nn.LeakyReLU(0.1)
        #     else:
        #         self.act = nn.ReLU(inplace=True)

            # self.conv_dr_block = nn.Sequential( #做降维
            # nn.Conv2d(self.input_dim, self.dr, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(self.dr),
            # self.act
            # )
        # output_dim = self.dr if self.dr else self.input_dim
        # if self.is_vec:
        #     self.output_dim = int(output_dim*(output_dim+1)/2)
        # else:
        #     self.output_dim = int(output_dim*output_dim)

        # self.temperature = nn.Parameter(torch.log((1. / (2 * input_dim[1]*input_dim[2])) * torch.ones(1,1)), requires_grad=True)

        # self._init_weight()

    # def _init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def BDCovpool(self, x): #, t
        batchSize, dim, h, w = x.data.shape
        M = h * w
        x = x.reshape(batchSize, dim, M)

        I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)
        I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)
        x_pow2 = x.bmm(x.transpose(1, 2))
        dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
        
        dcov = torch.clamp(dcov, min=0.0)
        # dcov = torch.exp(t)* dcov
        dcov = torch.sqrt(dcov + 1e-5)
        t = dcov - 1. / dim * dcov.bmm(I_M) - 1. / dim * I_M.bmm(dcov) + 1. / (dim * dim) * I_M.bmm(dcov).bmm(I_M)

        return t


    # def Triuvec(self, x):
    #     batchSize, dim, dim = x.shape
    #     r = x.reshape(batchSize, dim * dim)
    #     I = torch.ones(dim, dim).triu().reshape(dim * dim)
    #     index = I.nonzero(as_tuple = False)
    #     y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    #     y = r[:, index].squeeze()
    #     return y

    # def BDCblock(self, x):
        # if self.dr is not None and self.dr != self.input_dim:
        #     x = self.conv_dr_block(x)
        # x = self.BDCovpool(x, self.temperature)
        # if self.is_vec:
        #     x = self.Triuvec(x)
        # else:
        #     x = x.reshape(x.shape[0], -1)
        # return x

    # def cal_loss(self, f_s, f_t):
    #     f_s = F.normalize(f_s, dim=1)
    #     f_t = F.normalize(f_t, dim=1)
    #     dbc_loss = F.l1_loss(self.BDCovpool(f_s), self.BDCovpool(f_t))
    #     return dbc_loss

    def cal_loss(self, f_s, f_t):
        p_s = F.log_softmax(f_s/self.T, dim=1)
        p_t = F.softmax(f_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / f_t.shape[0]
        return loss

    def forward(self, g_s, g_t):
        return sum(self.cal_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t))


