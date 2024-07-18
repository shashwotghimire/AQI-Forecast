import pickle
import keras
modelPath = r'C:\Users\xd\Desktop\project\AQI-Forecast\ml_model\ML_model\lstm.keras'
modelPath2 = r'C:\Users\xd\Desktop\project\AQI-Forecast\ml_model\model1'
model = keras.models.load_model(modelPath2)    
# with open(modelPath, 'rb') as file:
#     loaded_model = pickle.load(file)
