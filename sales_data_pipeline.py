from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.kafka.operators.produce import ProduceToTopicOperator
from airflow.models import Variable
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError
from collections import defaultdict
import logging
import pandas as pd
import io
import pyodbc
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import wraps
import random

# Constants
KAFKA_TOPIC = 'sales_data'
BLOB_CONTAINER_NAME = "sales-data"

class Config:
    """Centralized configuration management"""
    def __init__(self):
        self.SHIP_MODES = ['Standard Class', 'Second Class', 'Same Day', 'First Class']
        self.SEGMENTS = ['Consumer', 'Home Office', 'Corporate']
        self.STATES = ['Constantine', 'New South Wales', 'Budapest', 'Karaman', 'Sikasso', 'Atsimo-Andrefana']
        self.COUNTRIES = ['Algeria', 'Australia', 'Hungary', 'Sweden', 'Canada', 'New Zealand', 'Iraq', 
                       'Philippines', 'United Kingdom', 'Malaysia', 'United States', 'Japan']
        self.MARKETS = ['Africa', 'APAC', 'EMEA', 'EU', 'Canada', 'LATAM', 'US']
        self.REGIONS = ['Africa', 'Oceania', 'EMEA', 'North', 'Canada', 'Southeast Asia', 'Central', 
                      'South', 'Caribbean', 'North Asia', 'East', 'West', 'Central Asia']
        self.CATEGORIES = ['Office Supplies', 'Furniture', 'Technology']
        self.SUB_CATEGORIES = {
            'Office Supplies': ['Storage', 'Supplies', 'Paper', 'Art', 'Envelopes', 'Fasteners', 'Binders', 'Labels'],
            'Furniture': ['Furnishings', 'Chairs', 'Tables', 'Bookcases'],
            'Technology': ['Machines', 'Appliances', 'Copiers', 'Phones', 'Accessories']
        }
        self.PRODUCT_NAMES = {
            ('Office Supplies', 'Storage'): ['File Box', 'Storage Drawer', 'Magazine Rack'],
            ('Office Supplies', 'Paper'): ['Printer Paper', 'Copy Paper', 'Note Pad'],
            ('Furniture', 'Chairs'): ['Executive Chair', 'Task Chair', 'Folding Chair'],
            ('Technology', 'Phones'): ['iPhone', 'Samsung Galaxy', 'Google Pixel']
        }
        self.ORDER_PRIORITIES = ['Medium', 'High', 'Critical', 'Low']

class AzureConnectionManager:
    """Manages Azure connections and operations"""
    def __init__(self):
        self.connection_string = Variable.get("AZURE_STORAGE_CONNECTION_STRING")
        self._blob_service_client = None
        self._container_client = None

    @property
    def blob_service_client(self):
        if not self._blob_service_client:
            self._blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        return self._blob_service_client

    @property
    def container_client(self):
        if not self._container_client:
            self._container_client = self.blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
        return self._container_client

    def get_blob_client(self, blob_name):
        return self.container_client.get_blob_client(blob_name)

class OrderTracker:
    """Tracks and generates unique order IDs"""
    def __init__(self):
        self._used_order_ids = defaultdict(set)
    
    def generate_unique_order_id(self, year):
        while True:
            order_id = f"{year}-{random.randint(1000, 9999)}"
            if order_id not in self._used_order_ids[year]:
                self._used_order_ids[year].add(order_id)
                return order_id

def log_errors(func):
    """Error logging decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

class SalesDataGenerator:
    """Handles sales data generation and processing"""
    def __init__(self):
        self.config = Config()
        self.order_tracker = OrderTracker()
        self.azure_manager = AzureConnectionManager()

    def generate_product_id(self):
        return f"P{random.randint(100000, 999999)}"

    def generate_sales_record(self):
        year = random.randint(2011, 2014)
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        order_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        ship_date = order_date + timedelta(days=random.randint(2, 5))
        
        category = random.choice(self.config.CATEGORIES)
        sub_category = random.choice(self.config.SUB_CATEGORIES[category])
        
        quantity = random.randint(1, 10)
        unit_price = random.uniform(10, 1000)
        discount = round(random.uniform(0, 0.5), 2)
        sales = round(quantity * unit_price * (1 - discount), 2)
        profit = round(sales * random.uniform(0.1, 0.3), 2)
        shipping_cost = round(random.uniform(5, 50), 2)

        return {
            'order_id': self.order_tracker.generate_unique_order_id(year),
            'order_date': order_date.strftime('%Y-%m-%d'),
            'ship_date': ship_date.strftime('%Y-%m-%d'),
            'ship_mode': random.choice(self.config.SHIP_MODES),
            'segment': random.choice(self.config.SEGMENTS),
            'state': random.choice(self.config.STATES),
            'country': random.choice(self.config.COUNTRIES),
            'market': random.choice(self.config.MARKETS),
            'region': random.choice(self.config.REGIONS),
            'product_id': self.generate_product_id(),
            'category': category,
            'sub_category': sub_category,
            'product_name': random.choice(self.config.PRODUCT_NAMES.get((category, sub_category), ['Generic Product'])),
            'sales': sales,
            'quantity': quantity,
            'discount': discount,
            'profit': profit,
            'shipping_cost': shipping_cost,
            'order_priority': random.choice(self.config.ORDER_PRIORITIES),
            'year': year
        }

    @log_errors
    def generate_batch_records(self, num_records=50):
        return [self.generate_sales_record() for _ in range(num_records)]

class DataProcessor:
    """Handles data processing and storage operations"""
    def __init__(self):
        self.azure_manager = AzureConnectionManager()

    @log_errors
    def generate_and_save_to_blob(self, **context):
        generator = SalesDataGenerator()
        records = generator.generate_batch_records(num_records=100)
        df = pd.DataFrame(records)
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        timestamp = context['ts_nodash']
        temp_blob_name = f'temp/sales_data_{timestamp}.csv'
        blob_client = self.azure_manager.get_blob_client(temp_blob_name)
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)
                logging.info(f"Successfully uploaded blob: {temp_blob_name}")
                return temp_blob_name
            except AzureError as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise
                logging.warning(f"Retry {retry_count} of {max_retries} due to: {str(e)}")

    @log_errors
    def kafka_producer_from_blob(self, blob_name):
        blob_client = self.azure_manager.get_blob_client(blob_name)
        data = blob_client.download_blob().readall()
        if isinstance(data, str):
            data = data.encode('utf-8')
        return [(None, data)]

    @log_errors
    def save_to_final_blob_location(self, **context):
        blob_name = context['task_instance'].xcom_pull(task_ids='generate_data')
        if not blob_name:
            raise ValueError("No blob name found in XCom")
        
        temp_blob_client = self.azure_manager.get_blob_client(blob_name)
        new_data = temp_blob_client.download_blob().readall().decode('utf-8')
        
        final_blob_name = 'consolidated_sales_data.csv'
        final_blob_client = self.azure_manager.get_blob_client(final_blob_name)
        
        new_df = pd.read_csv(io.StringIO(new_data))
        
        try:
            existing_data = final_blob_client.download_blob().readall().decode('utf-8')
            existing_df = pd.read_csv(io.StringIO(existing_data))
            combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['order_id'], keep='last')
        except Exception:
            combined_df = new_df
        
        combined_data = combined_df.to_csv(index=False)
        final_blob_client.upload_blob(combined_data, overwrite=True)

    @log_errors
    def cleanup_temp_file(self, **context):
        temp_blobs = self.azure_manager.container_client.list_blobs(name_starts_with='temp/')
        for blob in temp_blobs:
            logging.info(f"Attempting to delete temp file: {blob.name}")
            blob_client = self.azure_manager.get_blob_client(blob.name)
            blob_client.delete_blob()
            logging.info(f"Successfully deleted temp file: {blob.name}")

class SynapseProcessor:
    """Handles Synapse database operations"""
    def __init__(self):
        self.workspace_name = Variable.get("SYNAPSE_WORKSPACE_NAME")
        self.database_name = Variable.get("SYNAPSE_DATABASE_NAME")
        self.client_id = Variable.get("AZURE_CLIENT_ID")
        self.client_secret = Variable.get("AZURE_CLIENT_SECRET")
        self.tenant_id = Variable.get("AZURE_TENANT_ID")

    @property
    def connection_string(self):
        return (
            f"Driver={{ODBC Driver 18 for SQL Server}};"
            f"Server=tcp:{self.workspace_name}.sql.azuresynapse.net,1433;"
            f"Database={self.database_name};"
            f"Authentication=ActiveDirectoryServicePrincipal;"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"UID={self.client_id}@{self.tenant_id};"
            f"PWD={self.client_secret};"
            f"Tenant={self.tenant_id};"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def execute_sql_commands(self, sql_commands):
        with pyodbc.connect(self.connection_string, autocommit=True) as conn:
            with conn.cursor() as cursor:
                for sql in sql_commands:
                    logging.info(f"Executing SQL: {sql[:100]}...")
                    cursor.execute(sql)

    @log_errors
    def copy_to_synapse(self, **context):
        sql_commands = [
            """
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'sales_data')
            CREATE TABLE [dbo].[sales_data]
            (
                order_id VARCHAR(20) NOT NULL,
                order_date DATE NOT NULL,
                ship_date DATE NOT NULL,
                ship_mode VARCHAR(50) NOT NULL,
                segment VARCHAR(50) NOT NULL,
                state VARCHAR(100) NOT NULL,
                country VARCHAR(100) NOT NULL,
                market VARCHAR(50) NOT NULL,
                region VARCHAR(50) NOT NULL,
                product_id VARCHAR(20) NOT NULL,
                category VARCHAR(50) NOT NULL,
                sub_category VARCHAR(50) NOT NULL,
                product_name VARCHAR(255) NOT NULL,
                sales DECIMAL(18,2) NOT NULL,
                quantity INT NOT NULL,
                discount DECIMAL(5,2) NOT NULL,
                profit DECIMAL(18,2) NOT NULL,
                shipping_cost DECIMAL(18,2) NOT NULL,
                order_priority VARCHAR(20) NOT NULL,
                year INT NOT NULL
            )
            WITH
            (
                DISTRIBUTION = HASH(order_id),
                CLUSTERED COLUMNSTORE INDEX
            );
            """,
            """
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'sales_data_staging')
            CREATE TABLE [dbo].[sales_data_staging]
            (
                order_id VARCHAR(20),
                order_date DATE,
                ship_date DATE,
                ship_mode VARCHAR(50),
                segment VARCHAR(50),
                state VARCHAR(100),
                country VARCHAR(100),
                market VARCHAR(50),
                region VARCHAR(50),
                product_id VARCHAR(20),
                category VARCHAR(50),
                sub_category VARCHAR(50),
                product_name VARCHAR(255),
                sales DECIMAL(18,2),
                quantity INT,
                discount DECIMAL(5,2),
                profit DECIMAL(18,2),
                shipping_cost DECIMAL(18,2),
                order_priority VARCHAR(20),
                year INT
            )
            WITH
            (
                DISTRIBUTION = ROUND_ROBIN,
                HEAP
            );
            """,
            "TRUNCATE TABLE [dbo].[sales_data_staging];",
            """
            DELETE FROM t1
            FROM [dbo].[sales_data] t1
            JOIN (
                SELECT order_id, MAX(order_date) as max_date
                FROM [dbo].[sales_data]
                GROUP BY order_id
                HAVING COUNT(*) > 1
            ) t2 ON t1.order_id = t2.order_id
            WHERE t1.order_date < t2.max_date;
            """,
            """
            COPY INTO [dbo].[sales_data_staging]
            FROM 'https://salesprojectstore.blob.core.windows.net/sales-data/consolidated_sales_data.csv'
            WITH (
                FILE_TYPE = 'CSV',
                FIRSTROW = 2,
                FIELDTERMINATOR = ','
            );
            """,
            """
            INSERT INTO [dbo].[sales_data]
            SELECT s.*
            FROM [dbo].[sales_data_staging] s
            LEFT JOIN [dbo].[sales_data] t
                ON s.order_id = t.order_id
            WHERE t.order_id IS NULL;
            """,
            "TRUNCATE TABLE [dbo].[sales_data_staging];",
             """
            IF OBJECT_ID('dbo.sales_data_staging', 'U') IS NOT NULL 
                DROP TABLE [dbo].[sales_data_staging];
            """
        ]
        self.execute_sql_commands(sql_commands)


# DAG definition
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=15),
}

with DAG(
    'sales_data_generator',
    default_args=default_args,
    description='Generate sales data and push to Kafka and Azure Blob Storage in CSV format',
    schedule_interval='*/10 * * * *',
    catchup=False
) as dag:
    
    processor = DataProcessor()
    synapse_processor = SynapseProcessor()

    generate_data = PythonOperator(
        task_id='generate_data',
        python_callable=processor.generate_and_save_to_blob,
        provide_context=True,
    )

    produce_to_kafka = ProduceToTopicOperator(
        task_id='produce_to_kafka',
        kafka_config_id="kafka_default",
        topic=KAFKA_TOPIC,
        producer_function=processor.kafka_producer_from_blob,
        producer_function_args=["{{ task_instance.xcom_pull(task_ids='generate_data') }}"],
    )

    save_records = PythonOperator(
        task_id='save_records',
        python_callable=processor.save_to_final_blob_location,
        provide_context=True,
    )

    synapse_copy = PythonOperator(
        task_id='copy_to_synapse',
        python_callable=synapse_processor.copy_to_synapse,
        provide_context=True,
    )

    cleanup = PythonOperator(
        task_id='cleanup',
        python_callable=processor.cleanup_temp_file,
        provide_context=True,
    )

    generate_data >> produce_to_kafka >> save_records >> synapse_copy >> cleanup