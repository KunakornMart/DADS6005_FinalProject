import os
import json
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mongo.hooks.mongo import MongoHook
from datetime import datetime, timedelta
import pandas as pd
import pendulum
from prophet import Prophet
import pickle
import time
from bson import ObjectId

def on_failure_callback(**context):
    print(f"Task {context['task_instance_key_str']} failed.")

def downloadFromMongo(ti, **kwargs):
    try:
        hook = MongoHook(mongo_conn_id='mongo_default')
        client = hook.get_conn()
        db = client.MyDB
        btc_collection = db.btc_collection
        print(f"Connected to MongoDB - {client.server_info()}")

        # Convert ObjectId to string
        def convert_object_id(data):
            if isinstance(data, list):
                return [convert_object_id(item) for item in data]
            elif isinstance(data, dict):
                return {key: convert_object_id(value) for key, value in data.items()}
            elif isinstance(data, ObjectId):
                return str(data)
            else:
                return data

        # Fetch data from MongoDB
        raw_data = list(btc_collection.find())
        print(f"Raw data fetched from MongoDB: {raw_data}")  # Debugging line
        
        if not raw_data:
            print("No data found in MongoDB.")
            ti.xcom_push(key="data", value=None)  # Push None if no data found
            return

        # Convert ObjectId and prepare data for XCom
        converted_data = convert_object_id(raw_data)
        results = json.dumps(converted_data)
        print(f"Converted data: {results}")

        ti.xcom_push(key="data", value=results)  # Use 'ti' to push to XCom

    except Exception as e:
        print(f"Error connecting to MongoDB -- {e}")
        ti.xcom_push(key="data", value=None)  # Push None in case of an error

def trainModel(ti, **kwargs):
    data = ti.xcom_pull(
        task_ids="Get_BTC_from_Mongodb",  # Correct task_id
        key="data"
    )
    
    if data is None:
        print("Error: No data received from MongoDB.")
        return  # Exit the function if no data

    print("Get data for training:", data)
    
    # Convert the JSON data into DataFrame
    df = pd.DataFrame(json.loads(data))
    print(f"DataFrame head:\n{df.head()}")
    
    # Prepare the data for Prophet model (Prophet requires columns 'ds' and 'y')
    df['ds'] = pd.to_datetime(df['closeTime'], unit='ms')  # Convert closeTime from milliseconds to datetime
    df['y'] = df['lastPrice']
    df = df[['ds', 'y']]  # Prophet requires the data in this format
    
    # Initialize and train the Prophet model with adjusted hyperparameters
    model = Prophet(
        changepoint_prior_scale=0.1,           # ปรับความไวในการจับการเปลี่ยนแปลงในแนวโน้ม
        seasonality_prior_scale=10.0,          # ปรับความไวในการจับฤดูกาล
        holidays_prior_scale=10.0,             # ปรับความไวในการจับวันหยุดหรือเหตุการณ์พิเศษ
        seasonality_mode='multiplicative',     # ใช้ multiplicative seasonality (ฤดูกาลที่มีการขยายตามการเติบโต)
        yearly_seasonality=True,               # ใช้ seasonality รายปี
        weekly_seasonality=False,              # ไม่ใช้ seasonality รายสัปดาห์ (ถ้าไม่มี)
        interval_width=0.95,                   # เพิ่มช่วงความเชื่อมั่นเป็น 95%
    )

    # Fit the model with the training data
    model.fit(df)

    # Save the model
    print("Path at terminal when executing this file")
    print(os.getcwd() + "\n")
    
    modelFile = '/usr/local/airflow/tests/model_prophet.pkl'
    print(f"Saving model to: {modelFile}")
    
    with open(modelFile, 'wb') as f:
        pickle.dump(model, f)


local_tz = pendulum.timezone("Asia/Bangkok")
with DAG(
    dag_id="Train_ML_Model_Prophet",
    schedule_interval="4-59/5 * * * *",  # Adjust to every 1 hour
    start_date=datetime(2024, 12, 21, tzinfo=local_tz),
    catchup=False,
    tags=["crypto"],
    default_args={
        "owner": "Ekarat_Kunakorn",
        "retries": 2,
        "retry_delay": timedelta(minutes=1),
        "on_failure_callback": on_failure_callback
    }
) as dag:

    t1 = PythonOperator(
        task_id="Get_BTC_from_Mongodb",
        python_callable=downloadFromMongo,
        provide_context=True,  # Enable context
    )

    t2 = PythonOperator(
        task_id="train_model_with_prophet",
        python_callable=trainModel,
        provide_context=True,  # Enable context
    )

    t1 >> t2
