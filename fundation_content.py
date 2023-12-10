(1)断点续训存取模型：采用checkpoint的方法，调用tf.keras.callbacks.ModelCheckpoint，然后在model.fit中加入callbacks=[cp_callback]；调用的时候还是需要重新构建模型，然后采用model.load_weight来调用checkpoint的内容
        import tensorflow as tf
        import os
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        # 读取模型load_weights(路径文件名)
        checkpoint_save_path = "./checkpoint/mnist.ckpt"# 生成ckpt文件的时候，会产生相应的索引表
        if os.path.exists(checkpoint_save_path + '.index'):# 通过判断是否有索引表，去判断是否保存过模型的参数
            print('-------------load the model-----------------')
            model.load_weights(checkpoint_save_path)# 读取模型参数，这里就是断点续训
        # 保存模型： 
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                         save_weights_only=True,# 是否只保留模型参数
                                                         save_best_only=True)# 是否只保留最有结果
        history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                            callbacks=[cp_callback])
        model.summary()
(2)训练参数的可视化：采用history=model.fit(),然后调用acc=history.history['列名’]来进行调用对应的损失行数参数，plt.plot（loss， label=‘’）
        import matplotlib.pyplot as plt
        # 显示训练集和验证集的acc和loss曲线
        acc = history.history['sparse_categorical_accuracy']
        val_acc = history.history['val_sparse_categorical_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
（3）模型的加载和预测：model.load_weight和model.predict
        import tensorflow as tf
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')])
        model.load_weights(model_save_path)
        result=model.predict(x_predict)
(4)图像处理博客的网络地址：https://betterbench.blog.csdn.net/article/details/109519282
   卷积--标准化--池化
        积输出维度计算已知：假设输入的图片 尺寸 ：A x A     卷积核 大小：K    步长Stride：S   Padding大小：P    求解： 输出的embedding 维度 B：B = (A + 2*P - K) / S + 1
        
        tf.keras.layers.Conv2D(
        filters = 卷积核个数，
        kernel_size = 卷积核尺寸，# 正方形写核长整数，或（核高h,核宽w）
        strides = 滑动步长，#横纵向相同写步长整数，或（纵向步长h，横向步长w）,默认1
        padding = “same” or “valid” .#使用全零填充是“same”，不适用是“valid”（默认）
        activation = “relu” or “sigmoid” or “tanh” or "softmax"等，#如有BN（批标准化操作）此处不谢
        input_shape = （高，宽，通道数）# 输入特征图维度，可省略)

        model = tf.keras.models.Sequential([
            Conv2D(filters = 6,kernel_size = (5,5),padding = 'same'),#卷积层
            BatchNormalization(),#BN层
            Activation（'relu'）,
            MaxPool2D(pool_size =(2,2),strides = 2,padding = 'same'),
            Dropout(0.2),# dropout层
        ])


        
