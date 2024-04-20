# Basic Structure:
# Start with VGG16 or other CNN
# Additional layers: On top of CNN, add more convolution with dilation


# Implementation for SSD net
def _built_net(self):
    """Construct the SSD net"""
    self.end_points = {}  # record the detection layers output
    self._images = tf.placeholder(tf.float32, shape=[None, self.ssd_params.img_shape[0],
                                                     self.ssd_params.img_shape[1], 3])
    with tf.variable_scope("ssd_300_vgg"):
        # original vgg layers
        # block 1
        net = conv2d(self._images, 64, 3, scope="conv1_1")
        net = conv2d(net, 64, 3, scope="conv1_2")
        self.end_points["block1"] = net
        net = max_pool2d(net, 2, scope="pool1")
        # block 2
        net = conv2d(net, 128, 3, scope="conv2_1")
        net = conv2d(net, 128, 3, scope="conv2_2")
        self.end_points["block2"] = net
        net = max_pool2d(net, 2, scope="pool2")
        # block 3
        net = conv2d(net, 256, 3, scope="conv3_1")
        net = conv2d(net, 256, 3, scope="conv3_2")
        net = conv2d(net, 256, 3, scope="conv3_3")
        self.end_points["block3"] = net
        net = max_pool2d(net, 2, scope="pool3")
        # block 4
        net = conv2d(net, 512, 3, scope="conv4_1")
        net = conv2d(net, 512, 3, scope="conv4_2")
        net = conv2d(net, 512, 3, scope="conv4_3")
        self.end_points["block4"] = net
        net = max_pool2d(net, 2, scope="pool4")
        # block 5
        net = conv2d(net, 512, 3, scope="conv5_1")
        net = conv2d(net, 512, 3, scope="conv5_2")
        net = conv2d(net, 512, 3, scope="conv5_3")
        self.end_points["block5"] = net
        print(net)
        net = max_pool2d(net, 3, stride=1, scope="pool5")
        print(net)

        # additional SSD layers
        # block 6: use dilate conv
        net = conv2d(net, 1024, 3, dilation_rate=6, scope="conv6")
        self.end_points["block6"] = net
        #net = dropout(net, is_training=self.is_training)
        # block 7
        net = conv2d(net, 1024, 1, scope="conv7")
        self.end_points["block7"] = net
        # block 8
        net = conv2d(net, 256, 1, scope="conv8_1x1")
        net = conv2d(pad2d(net, 1), 512, 3, stride=2, scope="conv8_3x3",
                     padding="valid")
        self.end_points["block8"] = net
        # block 9
        net = conv2d(net, 128, 1, scope="conv9_1x1")
        net = conv2d(pad2d(net, 1), 256, 3, stride=2, scope="conv9_3x3",
                     padding="valid")
        self.end_points["block9"] = net
        # block 10
        net = conv2d(net, 128, 1, scope="conv10_1x1")
        net = conv2d(net, 256, 3, scope="conv10_3x3", padding="valid")
        self.end_points["block10"] = net
        # block 11
        net = conv2d(net, 128, 1, scope="conv11_1x1")
        net = conv2d(net, 256, 3, scope="conv11_3x3", padding="valid")
        self.end_points["block11"] = net

        # class and location predictions
        predictions = []
        logits = []
        locations = []
        for i, layer in enumerate(self.ssd_params.feat_layers):
            cls, loc = ssd_multibox_layer(self.end_points[layer], self.ssd_params.num_classes,
                                          self.ssd_params.anchor_sizes[i],
                                          self.ssd_params.anchor_ratios[i],
                                          self.ssd_params.normalizations[i], scope=layer+"_box")
            predictions.append(tf.nn.softmax(cls))
            logits.append(cls)
            locations.append(loc)
        return predictions, logits, locations

# Get the class and location prediction from detection layer
def ssd_multibox_layer(x, num_classes, sizes, ratios, normalization=-1, scope="multibox"):
    pre_shape = x.get_shape().as_list()[1:-1]
    pre_shape = [-1] + pre_shape
    with tf.variable_scope(scope):
        # l2 norm
        if normalization > 0:
            x = l2norm(x, normalization)
            print(x)
        # numbers of anchors
        n_anchors = len(sizes) + len(ratios)
        # location predictions
        loc_pred = conv2d(x, n_anchors*4, 3, activation=None, scope="conv_loc")
        loc_pred = tf.reshape(loc_pred, pre_shape + [n_anchors, 4])
        # class prediction
        cls_pred = conv2d(x, n_anchors*num_classes, 3, activation=None, scope="conv_cls")
        cls_pred = tf.reshape(cls_pred, pre_shape + [n_anchors, num_classes])
        return cls_pred, loc_pred