import pandas as pd

def getDataset(path):
    return pd.read_csv(path, sep=',',encoding='cp1252')

#Info and head output
def info_head(dataset, n):
    dataset.info()
    print(dataset.head(n))

#Drop column list
def drop_columns(dataset, columns):
    for column in columns:
        dataset = dataset.drop(column, axis = 1)
    return dataset

#Remove rows with null in columns
def remove_null_rows_by_columns(dataset, columns):
    for column in columns:
        dataset = dataset[~dataset[column].isnull()]
    return dataset

#Convert column to int
def toInt(dataset,columns):
    for column in columns:
        dataset[column] = dataset[column].astype(int)

#Remove errors in country names
def clean_errors(dataset):
    dataset.replace('-\d*','',regex=True, inplace = True)
    print(dataset[dataset["ID"] == 705])

#Fix population dataset
def pop_fix(Pop):
    Pop_c = Pop.iloc[: , :2]
    Pop_v = Pop.iloc[:,-40:-3]
    for i in range(len(Pop_v)) :
        Pop_v.iloc[i] = Pop_v.iloc[i].fillna(value =  Pop_v.iloc[i].mean())
    Pop = pd.concat([Pop_c, Pop_v],axis=1)
    Pop = Pop[~Pop['2016'].isnull()]
    Pop = Pop.melt(id_vars=["Country Name", "Country Code"], var_name="Year", value_name="Population")
    toInt(Pop,["Year"])
    return Pop

#Fix gdp dataset
def gdp_fix(GDP):
    GDP_c = GDP.iloc[: , :2]
    GDP_v = GDP.iloc[:,-42:-5]
    for i in range(len(GDP_v)) :
        GDP_v.iloc[i] = GDP_v.iloc[i].fillna(value =  GDP_v.iloc[i].mean())
    GDP = pd.concat([GDP_c, GDP_v], axis=1)
    GDP = remove_null_rows_by_columns(GDP, ["2016"])
    GDP = GDP.melt(id_vars=["Country Name", "Code"], var_name="Year", value_name="GDP")
    toInt(GDP, ["Year"])
    return GDP

#Rename code of countries
def renameCode(dataset, column, lst1,lst2):
    for x in range(len(lst1)):
        dataset[column].replace(lst1[x],lst2[x],inplace = True) 

def main():
    dataset = getDataset('data/medals.csv')
    dataset = dataset[dataset["Year"]>=1980]
    info_head(dataset, 5)
    dataset = drop_columns(dataset, ["Games", "City", "Event", "Name"])
    dataset = remove_null_rows_by_columns(dataset, ["Age", "Height", "Weight"])
    toInt(dataset, ["Age", "Height", "Weight"])
    info_head(dataset, 5)
    clean_errors(dataset)
    GDP = getDataset('data/GDP.csv')
    GDP = gdp_fix(GDP)
    Pop = getDataset('data/population.csv')
    Pop = drop_columns(Pop, ["Indicator Name"])
    Pop = pop_fix(Pop)
    info_head(Pop, 5)
    lst1 = ["NED","IRI","ISV","BRU","CGO","GBS","BAH","VIN","SKN","GAM","GER","BUL","GRE","NCA","ALG","KUW",\
            "LIB","MAS","RSA","LBA","SUD","KSA","INA","UAE","SRI","NGR","LAT","SUI","CRC","SLO","CRO","POR",\
            "ANG","BAN","URU","PUR","HON","MRI","SEY","MTN","NIG","PHI","NEP","MGL","MON","ASA","TOG","SAM",\
            "HAI","DEN","GUI","BIZ","PAR","BER","TAN","OMA","FIJ","VAN","GUA","ESA","MAD","CHA","CAY","BAR",\
            "BOT","ANT","ZIM","GRN","MYA","MAW","TGA","GEQ","SOL","ARU","ZAM","CAM","BHU","VIE","KOS","LES","BUR"]
    lst2 = ["NLD","IRN","VIR","BRN","COG","GNB","BHS","VCT","KNA","GMB","DEU","BGR","GRC","NIC","DZA","KWT",\
            "LBN","MYS","ZAF","LBY","SDN","SAU","IDN","ARE","LKA","NGA","LVA","CHE","CRI","SVN","HRV","PRT",\
            "AGO","BGD","URY","PRI","HND","MUS","SYC","MRT","NER","PHL","NPL","MNG","MCO","ASM","TGO","WSM",\
            "HTI","DNK","GIN","BLZ","PRY","BMU","TZA","OMN","FJI","VUT","GTM","SLV","MDG","TCD","CYM","BRB",\
            "BWA","ATG","ZWE","GRD","MMR","MWI","TON","GNQ","SLB","ABW","ZMB","KHM","BTN","VNM","XKX","LSO","BFA"]
    renameCode(dataset, "NOC", lst1, lst2)
    dataset = pd.merge(dataset, GDP,  how='left', left_on=['NOC','Year'], right_on = ['Code','Year'])
    dataset = pd.merge(dataset, Pop,  how='left', left_on=['NOC','Year'], right_on = ['Country Code','Year'])
    dataset.info()
    remove_columns = ["Country Name_x", "Country Name_y", "Country Code", "NOC"]
    dataset = drop_columns(dataset, remove_columns)
    dataset = remove_null_rows_by_columns(dataset, ["GDP", "Population"])
    dataset.to_csv("FullDataset.csv", sep=',', encoding='utf-8')

if __name__ == "__main__":
    main()