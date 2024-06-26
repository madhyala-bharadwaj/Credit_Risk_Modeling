import pandas as pd
from numpy import seterr
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import logging
import os

pd.options.mode.chained_assignment = None  # Suppress the warning
seterr(divide='ignore')  # Ignore divide by zero warnings

# Set up logging
logging_str = "[%(asctime)s:]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "credit_risk_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

# Global variables
education_mapping = {
    'SSC': 1, '12TH': 2, 'GRADUATE': 3, 'UNDER GRADUATE': 3,
    'POST-GRADUATE': 4, 'OTHERS': 1, 'PROFESSIONAL': 3
}

def drop_rows_with_missing_vals(data):
    for col in data.columns:
        if data[col].dtype != 'object':
            data = data[data[col]!=-99999]
    return data

def remove_features(data):
    '''
    Removes features containing more than 20% of missing values
    '''
    thresh = 0.2*data.shape[0]
    cols = []
    for col in data.columns:
        if data[col].dtype != 'object':
            if data[data[col]==-99999].shape[0] > thresh:
                cols.append(col)
    if cols: logging.info('Features to be removed: ',cols)
    return data.drop(cols,axis=1)

# Data Reading and Cleaning
def read_and_clean_data(file_path1, file_path2):
    try:
        df1 = pd.read_excel(file_path1)
        df2 = pd.read_excel(file_path2)
        logging.info('Data loaded successfully.')
    except Exception as e:
        logging.error(f"Error reading data: {e}")
        return None
    try:
        df1 = remove_features(df1)
        df1 = drop_rows_with_missing_vals(df1)
        df2 = remove_features(df2)
        df2 = drop_rows_with_missing_vals(df2)

        df = pd.merge(df1, df2, how='inner', on='PROSPECTID')
        df.drop(['PROSPECTID'], axis=1, inplace=True)
        logging.info('Data Cleaned successfully.')
        return df
    except Exception as e:
        logging.error(f"Error Cleaning data: {e}")

def encode_education(df):
    df['EDUCATION'] = df['EDUCATION'].map(education_mapping).astype(int)
    return df

def one_hot_encode(df, columns):
    return pd.get_dummies(df, columns=columns)

# Feature Selection
def chi_squared_test(df, target_column,strictness_level=0.05):
    try:
        columns_dropped = []
        for col in df.columns:
            if df[col].dtype=='object' and col!=target_column:
                chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[col], df[target_column]))
                logging.info(f'Chi-squared test for {col} - p-value: {pval}')
                if pval>=strictness_level:
                    df.drop([col],axis=1,inplace=True)
                    columns_dropped.append(col)
        if columns_dropped: logging.info(f'Chi-squared test - Columns dropped: {columns_dropped}')
        return df
    except Exception as e:
        logging.error(f"Error in chi-squared test: {e}")

def reduce_vif(df, threshold=6):
    try:
        numeric_columns = [col for col in df.columns if df[col].dtype != 'object']
        vif_data = df[numeric_columns]
        columns_dropped = []
        column_index = 0
        total_cols = len(numeric_columns)
        for i in range(total_cols):
            vif_value = variance_inflation_factor(vif_data, column_index)
            if vif_value > threshold:
                vif_data.drop([numeric_columns[i]],axis=1,inplace=True)
                df.drop([numeric_columns[i]],axis=1,inplace=True)
                columns_dropped.append(numeric_columns[i])
            else: column_index += 1
        if columns_dropped: logging.info(f'Reduced VIF - Columns dropped: {columns_dropped}')
        return df
    except Exception as e:
        logging.error(f"Error in reducing VIF: {e}")

def anova_test(df, target_column,strictness_level=0.05):
    try:
        numeric_columns = [col for col in df.columns if df[col].dtype != 'object']
        columns_dropped = []
        for col in numeric_columns:
            groups = [df[df[target_column] == group][col].dropna() for group in df[target_column].unique()]
            f_stat, p_val = f_oneway(*groups)
            if p_val > strictness_level:
                df.drop(columns=[col],inplace=True)
                columns_dropped.append(col)
        if columns_dropped: logging.info(f'ANOVA test - Columns dropped: {columns_dropped}')
        return df
    except Exception as e:
        logging.error(f"Error in ANOVA test: {e}")

# Scaling
def scale_data(df, columns):
    try:
        scalers = {}
        for col in columns:
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))
            scalers[col] = scaler
        logging.info('Data scaling complete')
        return df,scalers
    except Exception as e:
        logging.error(f"Error in data scaling: {e}")
        return df,{}

# Model Training and Evaluation
def train_and_evaluate_model(df, class_names):
    try:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['Approved_Flag'])
        x = df.drop(columns=['Approved_Flag'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Cross-validation and hyperparameter tuning
        param_grid = {
            'colsample_bytree': [0.3, 0.5, 0.7],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 5],
            'alpha': [10, 100],
            'n_estimators': [100, 200]
        }
        grid_search = GridSearchCV(estimator=XGBClassifier(objective='multi:softmax', num_class=4), param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
        logging.info('Started Hyperparameter tuning')
        grid_search.fit(x_train, y_train)

        best_model = grid_search.best_estimator_
        logging.info('Best Model Parameters: %s', grid_search.best_params_)

        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
        logging.info(f'Accuracy: {accuracy:.2f}')
        for i, class_name in enumerate(class_names):
            logging.info(f'Class {class_name} - Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1 Score: {f1[i]:.2f}')

        joblib.dump(best_model, r'data\best_model.pkl')
        joblib.dump(label_encoder, r'data\label_encoder.pkl')
        logging.info('Model and Label Encoder saved successfully.')

        return best_model, label_encoder
    except Exception as e:
        logging.error(f"Error in model training: {e}")

# Predicting on New Data
def predicting_on_unseen_data(model, file_path, save_path,features,scalers,columns_to_scale,label_encoder):
    try:
        df_unseen = pd.read_excel(file_path)
        df_unseen = df_unseen[features]
        df_unseen = encode_education(df_unseen)
        df_encoded_unseen = one_hot_encode(df_unseen, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])
        for col in columns_to_scale:
            df_encoded_unseen[col] = scalers[col].transform(df_encoded_unseen[col].values.reshape(-1,1))
        
        y_pred_unseen = model.predict(df_encoded_unseen)
        df_unseen['Target_variable'] = label_encoder.inverse_transform(y_pred_unseen)
        df_unseen.to_excel(save_path, index=False)
        logging.info(f'Predictions on unseen data saved at {save_path}')

    except Exception as e:
        logging.error(f"Error applying model on unseen data: {e}")



if __name__ == "__main__":
    df = read_and_clean_data('data\case_study1.xlsx', 'data\case_study2.xlsx')
    if df is not None:

        # Feature Selection
        df = chi_squared_test(df, 'Approved_Flag')
        df = reduce_vif(df)
        df = anova_test(df, 'Approved_Flag')

        # Encoding
        df = encode_education(df)
        df_encoded = one_hot_encode(df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])

        # Scaling
        columns_to_scale = ['Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment', 
                        'max_recent_level_of_deliq', 'recent_level_of_deliq', 
                        'time_since_recent_enq', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr']
        df_encoded,scalers = scale_data(df_encoded, columns_to_scale)
        
        # Training
        model,label_encoder = train_and_evaluate_model(df_encoded,['P1', 'P2', 'P3', 'P4'])

        # Predicting on Unseen Data
        features = list(df.columns)
        features.pop()
        predicting_on_unseen_data(model,r'data\Unseen_Dataset.xlsx','final_predictions.xlsx',features,scalers,columns_to_scale,label_encoder)