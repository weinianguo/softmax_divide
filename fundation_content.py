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
            model.load_weights(checkpoint_save_path)# 读取模型参数
        # 保存模型： 
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                         save_weights_only=True,# 是否只保留模型参数
                                                         save_best_only=True)# 是否只保留最有结果
        history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                            callbacks=[cp_callback])
        model.summary()
