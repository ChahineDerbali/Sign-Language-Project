import pandas as pd

data = {
    "Name": ["Tom", "Nick", "John", "Tom", "Nick", "John", "Tom", "Nick", "John"],
    "Age": [20, 21, 19, 20, 21, 19, 20, 21, 19],
}

df = pd.DataFrame(data)


df.head(2)
