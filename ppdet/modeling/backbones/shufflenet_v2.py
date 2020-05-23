#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from collections import OrderedDict

import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register

__all__ = ['ShuffleNetV2']


@register
class ShuffleNetV2():
    def __init__(self,
                 conv_group_scale=1.0,
                 norm_decay=0.,
                 freeze_norm=True,
                 feature_maps=[3, 4, 5],
                 weight_prefix_name="",
                 ):
        self.scale = conv_group_scale
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.feature_maps = feature_maps
        self.prefix_name = weight_prefix_name

    def __call__(self, input):
        scale = self.scale
        stage_repeats = [4, 8, 4]

        if scale == 0.25:
            stage_out_channels = [-1, 24, 24, 48, 96, 512]
        elif scale == 0.33:
            stage_out_channels = [-1, 24, 32, 64, 128, 512]
        elif scale == 0.5:
            stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif scale == 1.0:
            stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif scale == 1.5:
            stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif scale == 2.0:
            stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise NotImplementedError("This scale size:[" + str(scale) +
                                      "] is not implemented!")
        #conv1

        input_channel = stage_out_channels[1]
        conv1 = self.conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=input_channel,
            padding=1,
            stride=2,
            name='stage1_conv')
        pool1 = fluid.layers.pool2d(
            input=conv1,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        conv = pool1
        # bottleneck sequences
        res_endpoints = []
        if 2 in self.feature_maps:
            res_endpoints.append(conv)
        for idxstage in range(len(stage_repeats)):
            numrepeat = stage_repeats[idxstage]
            output_channel = stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    conv = self.inverted_residual_unit(
                        input=conv,
                        num_filters=output_channel,
                        stride=2,
                        benchmodel=2,
                        name=str(idxstage + 2) + '_' + str(i + 1))
                else:
                    conv = self.inverted_residual_unit(
                        input=conv,
                        num_filters=output_channel,
                        stride=1,
                        benchmodel=1,
                        name=str(idxstage + 2) + '_' + str(i + 1))
            if idxstage + 3 in self.feature_maps:
                res_endpoints.append(conv)

        return OrderedDict([('sfnv2_{}'.format(self.feature_maps[idx]), feat)
                                        for idx, feat in enumerate(res_endpoints)])

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      num_groups=1,
                      use_cudnn=True,
                      if_act=True,
                      name=None):
        _name = self.prefix_name + name if self.prefix_name != '' else name
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(
                initializer=MSRA(), name=_name + '_weights'),
            bias_attr=False)
        out = int((input.shape[2] - 1) / float(stride) + 1)
        bn_name = _name + '_bn'
        bn_lr = 0.0 if self.freeze_norm else 1.0
        norm_decay = self.norm_decay
        param_attr_bn = ParamAttr(
                name=bn_name + "_scale",
                learning_rate=bn_lr,
                regularizer=L2Decay(norm_decay))
        bias_attr_bn = ParamAttr(
                name=bn_name + "_offset",
                learning_rate=bn_lr,
                regularizer=L2Decay(norm_decay))
        if if_act:
            output = fluid.layers.batch_norm(
                input=conv,
                act='relu',
                param_attr=param_attr_bn,
                bias_attr=bias_attr_bn,
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')
        else:
            output = fluid.layers.batch_norm(
                input=conv,
                param_attr=param_attr_bn,
                bias_attr=bias_attr_bn,
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')
        scale = fluid.framework._get_var(param_attr_bn.name)
        bias = fluid.framework._get_var(bias_attr_bn.name)
        if self.freeze_norm:
            scale.stop_gradient = True
            bias.stop_gradient = True
        return output

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.shape[0], x.shape[
            1], x.shape[2], x.shape[3]
        channels_per_group = num_channels // groups

        # reshape
        x = fluid.layers.reshape(
            x=x, shape=[batchsize, groups, channels_per_group, height, width])

        x = fluid.layers.transpose(x=x, perm=[0, 2, 1, 3, 4])

        # flatten
        x = fluid.layers.reshape(
            x=x, shape=[batchsize, num_channels, height, width])

        return x

    def inverted_residual_unit(self,
                               input,
                               num_filters,
                               stride,
                               benchmodel,
                               name=None):
        assert stride in [1, 2], \
            "supported stride are {} but your stride is {}".format([1,2], stride)

        oup_inc = num_filters // 2
        inp = input.shape[1]

        if benchmodel == 1:
            x1, x2 = fluid.layers.split(
                input,
                num_or_sections=[input.shape[1] // 2, input.shape[1] // 2],
                dim=1)

            conv_pw = self.conv_bn_layer(
                input=x2,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv1')

            conv_dw = self.conv_bn_layer(
                input=conv_pw,
                num_filters=oup_inc,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=oup_inc,
                if_act=False,
                use_cudnn=False,
                name='stage_' + name + '_conv2')

            conv_linear = self.conv_bn_layer(
                input=conv_dw,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv3')

            out = fluid.layers.concat([x1, conv_linear], axis=1)

        else:
            #branch1
            conv_dw_1 = self.conv_bn_layer(
                input=input,
                num_filters=inp,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=inp,
                if_act=False,
                use_cudnn=False,
                name='stage_' + name + '_conv4')

            conv_linear_1 = self.conv_bn_layer(
                input=conv_dw_1,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv5')

            #branch2
            conv_pw_2 = self.conv_bn_layer(
                input=input,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv1')

            conv_dw_2 = self.conv_bn_layer(
                input=conv_pw_2,
                num_filters=oup_inc,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=oup_inc,
                if_act=False,
                use_cudnn=False,
                name='stage_' + name + '_conv2')

            conv_linear_2 = self.conv_bn_layer(
                input=conv_dw_2,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv3')
            out = fluid.layers.concat([conv_linear_1, conv_linear_2], axis=1)

        # return self.channel_shuffle(out, 2)
        return fluid.layers.shuffle_channel(out, 2)


