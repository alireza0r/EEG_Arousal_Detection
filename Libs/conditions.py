import argparse

parser = argparse.ArgumentParser(description ='Load csv info for spliter')

parser.add_argument('--path', 
                    metavar='path', 
                    type=int, 
                    defult='/content/MyDrive/MyDrive/data/full_details_last_modified.csv')

args = parser.parse_args()

with open(args.path, 'r') as f:
  df = pd.read_csv(f, index_col=0)

# TEST
split1 = {'CT-GD':'control ', 'Group':'test', 'Type':'BCI', 'Level':'EASY'}
split2 = {'CT-GD':'control ', 'Group':'test', 'Type':'BCI', 'Level':'HARD'}

split3 = {'CT-GD':'GD', 'Group':'test', 'Type':'BCI', 'Level':'EASY'}
split4 = {'CT-GD':'GD', 'Group':'test', 'Type':'BCI', 'Level':'HARD'}

split5 = {'CT-GD':'control ', 'Group':'test', 'Type':'SHAM', 'Level':'EASY'}
split6 = {'CT-GD':'control ', 'Group':'test', 'Type':'SHAM', 'Level':'HARD'}

split7 = {'CT-GD':'GD', 'Group':'test', 'Type':'SHAM', 'Level':'EASY'}
split8 = {'CT-GD':'GD', 'Group':'test', 'Type':'SHAM', 'Level':'HARD'}

# Train
split11 = {'CT-GD':'control ', 'Group':'train1', 'Type':'', 'Level':'EASY'}
split22 = {'CT-GD':'control ', 'Group':'train1', 'Type':'', 'Level':'HARD'}

split33 = {'CT-GD':'GD', 'Group':'train1', 'Type':'', 'Level':'EASY'}
split44 = {'CT-GD':'GD', 'Group':'train1', 'Type':'', 'Level':'HARD'}

# Silance
split55 = {'CT-GD':'control ', 'Group':'train2', 'Type':'', 'Level':'EASY'}
split66 = {'CT-GD':'control ', 'Group':'train2', 'Type':'', 'Level':'HARD'}

split77 = {'CT-GD':'GD', 'Group':'train2', 'Type':'', 'Level':'EASY'}
split88 = {'CT-GD':'GD', 'Group':'train2', 'Type':'', 'Level':'HARD'}

df_s1 = csv_spliter(df, split1)
df_s2 = csv_spliter(df, split2)
df_s3 = csv_spliter(df, split3)
df_s4 = csv_spliter(df, split4)
df_s5 = csv_spliter(df, split5)
df_s6 = csv_spliter(df, split6)
df_s7 = csv_spliter(df, split7)
df_s8 = csv_spliter(df, split8)

df_s11 = csv_spliter(df, split11)
df_s22 = csv_spliter(df, split22)
df_s33 = csv_spliter(df, split33)
df_s44 = csv_spliter(df, split44)
df_s55 = csv_spliter(df, split55)
df_s66 = csv_spliter(df, split66)
df_s77 = csv_spliter(df, split77)
df_s88 = csv_spliter(df, split88)

split_list = [df_s1,
              df_s1,
              df_s2,
              df_s3,
              df_s4,
              df_s5,
              df_s6,
              df_s7,
              df_s8,

              df_s11,
              df_s22,
              df_s33,
              df_s44,
              df_s55,
              df_s66,
              df_s77,
              df_s88]

conditions = [split1,
              split2,
              split3,
              split4,
              split5,
              split6,
              split7,
              split8,
              split11,
              split22,
              split33,
              split44,
              split55,
              split66,
              split77,
              split88]

print('Spliter is loaded')
