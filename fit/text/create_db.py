import sqlite3

connect = sqlite3.connect('./dataset_db/text/base.db') #Database creation
cursor = connect.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS text_ai(
   id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
   text TEXT NOT NULL,
   class INT NOT NULL
   );
""")

cursor.execute('SELECT * FROM text_ai;')

result = cursor.fetchall()

for i in result:
    print(i)

print(len(result))

cursor.close()
connect.close()