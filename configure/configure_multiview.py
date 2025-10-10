def get_default_config(data_name):

    if data_name in ['Mfeat']:
        """The default configs."""
        return dict(

            seed=5,
            view=5,
            Autoencoder=dict(
                arch1=[216, 1024, 1024, 1024, 40],
                arch2=[76, 1024, 1024, 1024, 40],
                arch3=[64, 1024, 1024, 1024, 40],
                arch4=[6, 1024, 1024, 1024, 40],
                arch5=[240, 1024, 1024, 1024, 40],

                activations='relu',
                batchnorm=True,
            ),
            Inference=dict(
                shared_hidden=[128, 64],
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                arch3=[128, 256, 128],
                arch4=[128, 256, 128],
                arch5=[128, 256, 128],



        ),
            training=dict(
                lr=1.0e-4,
                start_inference=50,
                batch_size=256,
                epoch=200,
                alpha=10,
                lambda1=0.1,
                lambda2=10,
                class_num=10,
                kernel_mul=0.1,
                kernel_num=6,
                lambda3=5,#5
                lambda4=0.1,
            ),
        )

    else:
        raise Exception('Undefined data name')
