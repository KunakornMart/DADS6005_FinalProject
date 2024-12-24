import os
import json
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import time
import requests
import pendulum
from airflow.models import Variable
import pickle

# Callback function in case of failure
def on_failure_callback(**context):
    print(f"Task {context['task_instance_key_str']} failed.")

# Global variable for prediction length
prediction_len = 12  # จำนวนรอบที่ต้องการให้ทำการคำนวณ

def predictBTC(ti, **kwargs):
    counter = int(Variable.get("task_execution_counter", default_var=0))
    print("Counter in predictBTC (before increment):", counter)

    # Execute once per 'prediction_len' times
    if counter % prediction_len == 0:
        # Load the Prophet model
        model_file = '/usr/local/airflow/tests/model_prophet.pkl'
        print(model_file)

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Prophet model file not found: {model_file}")

        with open(model_file, 'rb') as f:
            clf = pickle.load(f)

        time_list = []
        current_timestamp_ms = int(time.time() * 1000)

        for i in range(1, prediction_len):
            current_timestamp_ms = current_timestamp_ms + (60 * i * 1000)
            time_list.append(current_timestamp_ms)

        # Create a DataFrame with closeTime
        df = pd.DataFrame({'closeTime': time_list})
        print("df = ", df.head())

        # Convert closeTime to datetime and rename columns to 'ds' and 'y' for Prophet
        df['ds'] = pd.to_datetime(df['closeTime'], unit='ms')  # Convert milliseconds to datetime
        df['y'] = None  # Add y column as a placeholder (for prediction)

        # Make predictions
        y_hat = clf.predict(df[['ds']])
        print("y_hat = ", y_hat)

        # Add predictions to DataFrame
        df['predicted'] = y_hat['yhat']

        # Save predictions to CSV
        output_file = '/usr/local/airflow/tests/out.csv'
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    # Increment the counter
    counter += 1
    if counter == prediction_len:  # Reset counter when it reaches prediction_len
        counter = 0
    print("Counter in predictBTC (after increment/reset):", counter)
    Variable.set("task_execution_counter", counter)

def evaluateModel(**kwargs):
    counter = int(Variable.get("task_execution_counter", default_var=0))
    print("Counter in evaluateModel (before evaluation):", counter)

    error_list = Variable.get("error_list", default_var="[]")
    error_list = json.loads(error_list)

    if counter != 0 and counter % prediction_len != 0:
        # Fetch the latest price from Binance API
        url = 'https://api.binance.com/api/v3/ticker?symbol=BTCUSDT'
        response = requests.get(url)
        data = response.json()
        print("Binance data:", data)

        # Check if the prediction file exists
        file_path = '/usr/local/airflow/tests/out.csv'
        if os.path.exists(file_path):
            # Read the CSV file
            df = pd.read_csv(file_path)
            print("CSV file loaded successfully.")

            # Check if counter-1 is a valid index
            if 0 <= counter - 1 < len(df):
                predicted_price = df.loc[counter-1, 'predicted']
                print(f"predictedPrice = {predicted_price}")
                print(f"lastPrice = {data['lastPrice']}")

                # Calculate error
                y_hat = float(predicted_price)
                y_actual = float(data['lastPrice'])
                error_list.append(abs(y_hat - y_actual) / y_actual)
                print("Updated error list:", error_list)
                Variable.set("error_list", json.dumps(error_list))
            else:
                print(f"Invalid index: {counter-1}, DataFrame length: {len(df)}.")
        else:
            print(f"File '{file_path}' does not exist.")
    elif counter % prediction_len == 0:  # Calculate and save MAPE when counter reaches prediction_len
        mape_path = '/usr/local/airflow/tests/mape.txt'
        if len(error_list) > 0:
            mape = (sum(error_list) / len(error_list)) * 100
            print('MAPE =', mape)

            # Adjust to Bangkok timezone
            execution_date = kwargs["execution_date"] + timedelta(hours=7)
            formatted_time = execution_date.strftime('%Y-%m-%d %H:%M:%S+00:00')
            print("Formatted time = ", formatted_time)

            # Write MAPE to file
            with open(mape_path, 'a') as output:
                output.write(f"{formatted_time}, {mape}\n")
                print(f"MAPE saved to {mape_path}")

            # Clear error list after saving MAPE
            error_list = []
            Variable.set("error_list", json.dumps(error_list))
        else:
            print("No errors to calculate MAPE.")

        # Reset counter to 0
        counter = 0
        print("Counter reset to 0 after writing MAPE.")
        Variable.set("task_execution_counter", counter)
    else:
        # Increment counter
        counter += 1
        Variable.set("task_execution_counter", counter)
    print("Counter in evaluateModel (after evaluation):", counter)

# Timezone setup for Bangkok
local_tz = pendulum.timezone("Asia/Bangkok")

with DAG(
    dag_id="Predict_BTC_Prophet",
    schedule_interval="*/5 * * * *",  # Run every 5 minutes
    start_date=datetime(2024, 12, 21, 0, 0, 0, tzinfo=local_tz),
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
        task_id="predictBTC",
        python_callable=predictBTC,
        provide_context=True,  # Enable context
    )

    t2 = PythonOperator(
        task_id="evaluateModel",
        python_callable=evaluateModel,
        provide_context=True,  # Enable context
    )

    t1 >> t2  # Define task order
