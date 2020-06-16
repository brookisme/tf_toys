from tensorflow.keras import layers


#
# MODELS
#
def segmentor(out_ch,kernel_size=3,out_kernel_size=1,channels=[32,64,128]):
    def _model(x):
        for ch in channels:
            x=cbr_block(ch,kernel_size=kernel_size)(x)
        x=cbr(out_ch,kernel_size=out_kernel_size,relu=False)(x)
        return layers.Softmax()(x)
    return _model



#
# BLOCKS
#
def cbr_block(
        filters,
        kernel_size=3,
        padding='same',
        seperable=False,
        batch_norm=True,
        depth=3,
        dilation_rate=1,
        output_strides=1,
        internal_relu=True,
        output_relu=True,
        residual=True,
        **kwargs):
    """conv-bn-relu"""
    def _block(x):
        if residual:
            res=cbr(
                filters=1,
                kernel_size=kernel_size,
                padding=padding,
                dilation_rate=dilation_rate,
                batch_norm=batch_norm,
                strides=output_strides,
                relu=False,
                **kwargs)(x)
        for _ in range(depth-1):
            x=cbr(
                filters=filters,
                kernel_size=kernel_size,
                padding=padding,
                dilation_rate=dilation_rate,
                batch_norm=batch_norm,
                relu=internal_relu,
                **kwargs)(x)
        x=cbr(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            dilation_rate=dilation_rate,
            batch_norm=batch_norm,
            strides=output_strides,
            relu=False,
            **kwargs)(x)
        if residual:
            x=layers.Add()([x, res])
        if output_relu:
            x=layers.ReLU()(x)
        return x
    return _block


def cbr(
        filters,
        kernel_size=3,
        padding='same',
        seperable=False,
        batch_norm=True,
        relu=True,
        **kwargs):
    """conv-bn-relu"""
    def _block(x):
        if seperable:
            _conv=layers.SeparableConv2D
        else:
            _conv=layers.Conv2D
        x=_conv(filters=filters,kernel_size=kernel_size,padding=padding,**kwargs)(x)
        if batch_norm:
            x=layers.BatchNormalization()(x)
        if relu:
            x=layers.ReLU()(x)
        return x
    return _block

