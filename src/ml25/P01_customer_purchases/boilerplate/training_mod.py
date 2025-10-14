# ML
import pandas as pd
from sklearn.metrics import classification_report

# Custom
from ml25.P01_customer_purchases.boilerplate.utils import setup_logger
from ml25.P01_customer_purchases.data_processing_clase_mod import read_train_data
from sklearn.model_selection import train_test_split


def run_training(X, y, classifier: str):
    logger = setup_logger(f"training_{classifier}")
    logger.info("HOLA!!")
    X, y = read_train_data()

    X_train, X_test, y_train, t_test = train_test_split(
        X,y, test_size = 0.44, random_state = 42
    )
    def run_training(X, y, classifier: str):
        logger = setup_logger(f"training_{classifier}")
        logger.info("HOLA!!")
        
        # Don't read data again if it's passed as parameters
        # X, y = read_train_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.44, random_state = 42
        )
        
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        
        # Create and train your model
        # Example: if classifier == 'random_forest':
        #     model = RandomForestClassifier()
        #     model.fit(X_train, y_train)
        
        # Validate model
        # y_pred = model.predict(X_test)
        # logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Save model
        # joblib.dump(model, f'models/{classifier}_model.pkl')
        
        # return model
    logger.info(f"y_train shape: {y_train.shape}")
    
    # 1.Separar en entrenamiento y validacion

    # 2.Entrenamiento del modelo
    # model = PurchaseModel(...)

    # 5.Validacion

    # 6. Guardar modelo


if __name__ == "__main__":
    X, y = read_train_data()
    # model = ...
    # run_training(X, y, model)
