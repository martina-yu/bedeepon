### Check Dataset ###
Before training dataset, we need to make sure if all of the datas in testing dataset have no overlap with training dataset. If there is any overlap, we need to remove them from the concatenated training dataset to ensure there is no data leaking. 

```
concatenated dataset
    |----> test (arbab, song, marquart, zcd) dataset
        |----> Yes
            |----> check train dataset (arbab, song, marquart, zcd)
        |----> No
            |----> 0
```

****
```
Outcomes: ABE(256584)
696  | test zcd ---- | 696 song train dataset
0    | test marquart
1063 | test song --- | 685 zcd train dataset | 378 song train dataset? | ---> 43 overlap in raw data
0    | test arbab
```
****
**Solution:**

Remove those dupulcated sequences in training dataset. We must keep the balance in testing dataset.


## Check for CBE

```
concatenated dataset
    |----> test (arbab, song, marquart, zcd) dataset
        |----> Yes
            |----> check train dataset (arbab, song, marquart, zcd)
        |----> No
            |----> 0
```

****
```
Outcomes: CBE(366877) --> after add unedited dataset CBE(383380)
2328  | test zcd ---- | 2864 song train dataset
0    | test marquart
1078 | test song --- | 252 zcd train dataset | 1051 song train dataset? | ---> 43 overlap in raw data
0    | test arbab
```