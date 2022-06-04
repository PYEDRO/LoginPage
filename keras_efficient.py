from statistics import mode
from typing import Iterator, List, Union, Tuple
from datetime import datetime
from xmlrpc.client import boolean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorboard

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import History

import tensorflow_addons as tfa






def get_callbacks(model_name:str) -> List[Union[TensorBoard,EarlyStopping,ModelCheckpoint]]:

    logdir = ('logs/scalars/'+model_name+'_'+datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=logdir)

    early_stopping_callback = EarlyStopping(monitor = 'val_mean_absolute_percentage_error',min_delta=1,patience=10,verbose=2,mode='min',restore_best_weights=True)

    model_checkpoint_callback = ModelCheckpoint(
        './data/models'+model_name+'.h5',monitor ='val_mean_absolute_percentage_error',
        verbose=0,
        save_best_only=True,
        mode ='min',
        save_freq='epoch'
    )
    return [tensorboard_callback,early_stopping_callback,model_checkpoint_callback]

def run_model(model_name:str,model_function:Model,lr:float,train_generator: Iterator,validation_generator:Iterator,test_generator:Iterator) ->History:
    '''
    Essa função é responsável por rodar um model keras
    parâmetro model_function: Recebe o keras model function exemplo small cnn(), pode ser trocada pela adapt_efficient_net()
    lr : learning rate
    '''
    callbacks = get_callbacks(model_name)
    model = model_function
    model.summary()
    plot_model(model,to_file=model_name + '.jpg',show_shapes=True)
    radam = tfa.optimizers.RectifiedAdam(learning_rate=lr)
    ranger = tfa.optimizers.Lookahead(radam,sync_period=6,slow_step_size=0.5)
    optimizer = ranger

    model.compile(optimizer=optimizer,loss='mean_absolute_error',metrics=[MeanAbsoluteError(),MeanAbsolutePercentageError()])
    history = model.fit(train_generator,epochs = 100,validation_data = validation_generator,callbacks=callbacks,workers = 6)
    model.evaluate(test_generator,callbacks=callbacks)
    return history


def visual_augmentations(data_generator:ImageDataGenerator,df:pd.DataFrame):
    '''
    Visualização da data augmentation, utiliza a grid de matplot 3x3
    '''
    series = df.iloc[2]
    df_augmentation_visualization = pd.concat([series, series], axis=1).transpose()
    iterator_visualizations = data_generator.flow_from_dataframe(
        dataframe=df_augmentation_visualization,
        x_col="image_location",
         y_col="DXA",
        class_mode="raw",
            target_size=(224, 224),
        batch_size=1,)
    
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        batch = next(iterator_visualizations)
        img = batch[0]
        img = img[0,:,:,:]
        plt.imshow(img)
    plt.show()
    plt.close()




def create_generators(df:pd.DataFrame,train:pd.DataFrame,val: pd.DataFrame,test: pd.DataFrame,visualize_augmentations:bool) ->Tuple[Iterator,Iterator,Iterator]:
    '''
    Utiliza os Pandas Dataframe para utilizar o Keras ImageDataGenerator.
    Também é possível ver as augmentations no dataset pelo parametro visualize augmentations
    '''
    train_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.75, 1),
        shear_range=0.1,
        zoom_range=[0.75, 1],
        horizontal_flip=True,
        validation_split=0.2,
    ) 
    validation_generator = ImageDataGenerator(
        rescale=1.0 / 255
    )
    test_generator = ImageDataGenerator(rescale=1.0 / 255)
    if visualize_augmentations == True:
        visual_augmentations(train_generator,df)
    #target size = tamanho esperado pela efficient net, para yolo talvez tenhamso que colocar 416x416 se utilizar a tiny
    train_generator = train_generator.flow_from_dataframe(dataframe=train,x_col='image_location',y_col='DXA',class_mode='raw',target_size=(224,224),batch_size=128)
    validation_generator = validation_generator.flow_from_dataframe(dataframe=val,x_col='image_location',y_col='DXA',class_mode='raw',target_size=(224,224),batch_size=128)
    test_generator = test_generator.flow_from_dataframe(dataframe=test,x_col='image_location',y_col='DXA',class_mode='raw',target_size=(224,224),batch_size=128)

    return train_generator,validation_generator,test_generator

def get_mean_baseline(train: pd.DataFrame,val: pd.DataFrame)->float:
    '''
    
    Calcula a métrica de MAE(mean absolut error) e MAPE (Mean Absolut percentage error):

    '''
    y_hat = train['DXA'].mean()
    val['y_hat'] = y_hat
    mae = MeanAbsoluteError()
    mae = mae(val['DXA'], val['y_hat']).numpy()
    mape = MeanAbsolutePercentageError()
    mape = mape(val['DXA'],val['y_hat']).numpy()
    print(mae)
    print(f'Mean baseline MAPE: {mape}')
    return mape

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    '''
    Split do data frame em treino teste e validação: Razão de 70,20,10

    Recebe um Dataframe e retorna uma Union de 3 dataframes

    '''

    train, val = train_test_split(df, test_size = 0.2, random_state = 1)
    train, test = train_test_split(train, test_size = 0.125, random_state = 1)
    print(f'shape train: {train.shape}')
    print(f'shape train: {val.shape}')
    print(f'shape train: {test.shape}')
    print('Descrivie statics:')
    print(train.describe())

    return train, val, test


def small_cnn() ->Sequential:
    '''
    Uma cnn com input de imagens 224x224x3
    '''
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))

    return model


def adapt_efficient_net() -> Model:
    inputs = layers.Input(
        shape=(224, 224, 3)
    )  
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="efficientnetb0_notop.h5")
    
    model.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.4
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(1, name="pred")(x)


    model = keras.Model(inputs, outputs, name="EfficientNet")

    return model

def plot_results(model_history_small_cnn: History, model_history_eff_net: History, mean_baseline: float):
  
    dict1 = {
        "MAPE": model_history_small_cnn.history["mean_absolute_percentage_error"],
        "type": "training",
        "model": "small_cnn",
    }
    dict2 = {
        "MAPE": model_history_small_cnn.history["val_mean_absolute_percentage_error"],
        "type": "validation",
        "model": "small_cnn",
    }
    dict3 = {
        "MAPE": model_history_eff_net.history["mean_absolute_percentage_error"],
        "type": "training",
        "model": "eff_net",
    }
    dict4 = {
        "MAPE": model_history_eff_net.history["val_mean_absolute_percentage_error"],
        "type": "validation",
        "model": "eff_net",
    }

    s1 = pd.DataFrame(dict1)
    s2 = pd.DataFrame(dict2)
    s3 = pd.DataFrame(dict3)
    s4 = pd.DataFrame(dict4)
    df = pd.concat([s1, s2, s3, s4], axis=0).reset_index()
    grid = sns.relplot(data=df, x=df["index"], y="MAPE", hue="model", col="type", kind="line", legend=False)
    grid.set(ylim=(20, 100))  # set the y-axis limit
    for ax in grid.axes.flat:
        ax.axhline(
            y=mean_baseline, color="lightcoral", linestyle="dashed"
        )  
        ax.set(xlabel="Epoch")
    labels = ["small_cnn", "eff_net", "mean_baseline"] 
    plt.legend(labels=labels)
    plt.savefig("training_validation.png")
    plt.show()





def run(small_sample = False):
    '''
    small_sample: parâmetro para teste do codigo.
    - True: Pega apenas as 1000 primeiras imagens
    '''
    df = pd.read_pickle('/home/lapisco01/Área de Trabalho/phd-navar-main/Perfeitos.pkl')
    # path para imagens
    df['image_location']= ("./data/processed_images/"+df['zpid']+'.png')
    if small_sample == True:
        df= df.iloc[0:1000]
    train,val,test = split_data(df)
    mean_baseline = get_mean_baseline(train, val)
    #mean_baseline = get_mean_baseline(train,val)
    train_generator, validation_generator, test_generator = create_generators(df=df,train=train,val=val,test=test,visualize_augmentations=True)

    small_cnn_history = run_model(model_name='small_cnn',model_function=small_cnn(),
    lr=0.001,train_generator=train_generator,validation_generator=validation_generator,test_generator=test_generator)
    eff_net_history = run_model(model_name='eff_net',model_function=adapt_efficient_net(),lr=0.5,train_generator=train_generator,test_generator=test_generator,validation_generator=validation_generator)
    plot_results(small_cnn_history,eff_net_history,mean_baseline)
if __name__ == '__main__':
    run()