import pyodbc
import pandas as pd
import logging


class database:
    def __init__(self):
        self.data = pd.DataFrame()

    def post(self, data, lable):
        logging.debug('Starting Write DB Query')
        conn_string = 'Driver={ODBC Driver 17 for SQL Server};Server=susmanserver;Database=Nest;UID=susman;PWD=Susman;'
        conn = pyodbc.connect(conn_string)
        query = "INSERT " \
                "INTO [Halite].[dbo].[training_data_v2] " \
                    "([timestamp]" \
                    ", [data]" \
                    ", [label])" \
                "VALUES " \
                    "(GETUTCDATE()" \
                    ",'"+str(data)+"'" \
                    ",'"+str(lable)+"')"

        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        logging.debug('Write DB Query Complete')

    def get(self):
        logging.debug('Starting Read DB Query')
        conn_string = 'Driver={ODBC Driver 17 for SQL Server};Server=susmanserver;Database=Nest;UID=susman;PWD=Susman;'
        conn = pyodbc.connect(conn_string)
        query = "EXEC [Halite].[dbo].[GrabAllData_v2]"

        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.commit()
        logging.debug('Read DB Query Complete')
        return results
