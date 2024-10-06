'''Project - MLD'''
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# importing libreries

#import logging
from datetime import datetime
#from tokenize import group
#from turtle import color
#from cv2 import normalize
#from cv2 import split
#import dask.array as np
import numpy as np
from math import sqrt
from gsheetsdb import connect
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
#from matplotlib.pyplot import axis, table
import matplotlib.pyplot as plt
#import dask.dataframe as pd
import pandas as pd
#from pyrsistent import optional
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
#from sklearn.decomposition import sparse_encode
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.plotting import autocorrelation_plot
from sklearn.pipeline import FeatureUnion, Pipeline
import streamlit as st
import streamlit.components.v1 as stc
from streamlit_autorefresh import st_autorefresh
import seaborn as sns
from seaborn_qqplot import pplot
from PIL import Image
import check_guss_norm as cgn
import normalize as norm
import scipy.stats as stats
import cicd
#import pipe_try

image = Image.open('MLDicon.jpg')

st.set_page_config(page_title="ML-dashboard", page_icon=image, initial_sidebar_state="expanded",
                   menu_items={
                       'Get Help': 'https://github.com/SUNNY11286/project_MLD_help',
                       'Report a bug': "https://github.com/SUNNY11286/project_MLD_help",
                       'About': "# This is part of project ML-D. \n This is an *testing version* web page!"
                   })
#st.image(image, caption='Sunrise by the mountains')
#count = st_autorefresh(interval=7200000, limit=100, key="fizzbuzzcounter")
#countl = st_autorefresh(interval=7200000, limit=30, key="counter")
# st.write(count)

st.title('JAKE \tML-Dashboard')
st.write('WARNING \tit is under development so might be unstable.')
print('ML Dashboard Begings')
st.write('For any Issues and queries you can go the Get Help or Report section in Menu')

# logging
#logger = logging.getLogger()

#""" function for feature engineering """


def fea(x):
    # data pre-processing
    #x = StandardScaler()
    # x=x.fit_transform(x)
    col = list(x.columns)

    scaler = RobustScaler()
    #print(x)
    st.write(x)
    robust_df = scaler.fit_transform(x)
    robust_df = pd.DataFrame(robust_df, columns=col)
    # print(robust_df)
    return robust_df


def modeler(type, target, X_t, X_v, y_t, y_v):
    # print(target)
    df = pd.DataFrame()
    model = 'None'
    preds = []
    rmse = 0.0
    r2 = 0.0
    acc = 0.0
    if type == 1:
        model = RandomForestRegressor(
            n_estimators=50, random_state=0, n_jobs=2)
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        rmse = sqrt(mean_squared_error(y_v, preds))
        r2 = r2_score(y_v, preds)
        mape = mean_absolute_percentage_error(preds, y_v)
        acc = round(100-mape.mean(), 2)
        df = pd.DataFrame(y_v)
        df['preds'] = list(preds)
        # print(type(y_v))
        df['error'] = list(abs(df['preds']-df[target]))
        print(df['preds'].count())

    elif type == 2:
        st.write('under development due to few errors / bugs')
    # pred_val.append(preds)
    # y_val.append(y_val)

    return df, model, preds, rmse, r2, acc


def show_graph(nw_dat, features, target):
    #pair_plot_fig = sns.pairplot(nw_dat, corner=True, hue=target)
    #plt.suptitle("Relathionships between numerical features by GENDER", weight='bold')
    # st.pyplot(pair_plot_fig)
    #""" Graph of features and datetime """
    '''
    graph = st.selectbox(
        "Choose plot",
        ("None","pairplot", "individual")
    )
    '''
    # if graph == 'pairplot':
    #agree = st.checkbox('Pairplot')

    # if agree:
    # st.write('Great!')
    with st.expander("pairplot"):
        st.write('Pair-Plot')
        with st.container():
            st.write('uncomment for final output. Commented till testing.....')
            pair_plot_fig = sns.pairplot(nw_dat, corner=True, hue=target)
            st.pyplot(pair_plot_fig)
    # if graph == 'individual':
    with st.expander("individual"):
        st.write('individual plots')
        with st.container():
            for col in features:
                fig = plt.figure()
                # print(nw_dat.head(5))
                plt.plot(nw_dat[col])
                plt.xticks(rotation=90)
                plt.xlabel('date')
                plt.ylabel(col)
                plt.title(f'no. of {col} against date/time')
                st.pyplot(fig)

    """ Graph of taraget and datetime """
    fig = plt.figure()
    # print(nw_dat.head(5))
    plt.plot(nw_dat[target])
    plt.xticks(rotation=90)
    plt.xlabel('date')
    plt.ylabel(target)
    plt.title(f'no. of {target} against date/time')
    st.pyplot(fig)

#""" function for error graph printing """


def err_graph(df, flag, type):
    print('------ enter error graph printing ------')

    erfig2 = plt.figure()
    print(df.head(5))
    plt.plot(df['error'])
    plt.xticks(rotation=90)
    plt.xlabel('date')
    plt.ylabel('error vals')
    plt.title('no. of errors against date/time of whole time line test case')
    st.pyplot(erfig2)

    print("---------- error predicting ------------")
    err_features = list(df.columns)
    err_target = 'error'
    err_features.remove(err_target)
    # if flag:
    # err_features.remove()
    split_time = int(len(df.index)*80/100)
    st.write('error split time: ', split_time)
    train = df[:split_time]
    test = df[split_time:]

    def err_model(type, df):  # sourcery skip: avoid-builtin-shadow
        # errdf2=pd.DataFrame()
        erres = pd.DataFrame(columns=['duration', 'rmse', 'r2'])
        #eres=pd.DataFrame(columns = ['duration','rmse', 'r2'])
        if flag:

            #""" whole time line is used """
            str = 'whole time line is used'
            X_t, X_v, y_t, y_v = train[err_features], test[err_features], train[err_target], test[err_target]
            errdf, model, preds, rmse, r2, acc = modeler(
                type, err_target, X_t, X_v, y_t, y_v)
            print('\n the model used is : ', model,
                  '\n the rmse for the model is : ', rmse, '\n the r2 score is : ', r2)
            # errdf2=errdf2.append(errdf)
            erres = erres.append(
                {'duration': str, 'rmse': rmse, 'r2': r2, 'accuracy': acc}, ignore_index=True)

            err_fig2 = plt.figure()
            print(errdf.head(5))
            plt.plot(errdf['error'])
            plt.xticks(rotation=90)
            plt.xlabel('date')
            plt.ylabel('error vals')
            plt.title(
                'training on error and predicting on errors against date/time of whole time line test case')
            st.pyplot(err_fig2)

        else:
            X_train, X_test, y_train, y_test = train_test_split(
                df[err_features], df[err_target], test_size=0.8, random_state=0)
            errdf, model, preds, rmse, r2, acc = modeler(
                err_target, X_train, X_test, y_train, y_test)
            erres = erres.append(
                {'duration': 'all', 'rmse': rmse, 'r2': r2, 'accuracy': acc}, ignore_index=True)

        st.write('result of train on error and predicting error')
        st.table(erres)

    err_model(type, df)

    # pass

#""" function for label encoder """


def label_encode(x):
    numerical_feature = []
    for i in x.columns:
        st.write(i)
        df_l = len(x[i].unique())
        st.write(df_l)
        #oh_e = OrdinalEncoder()
        if df_l < 7:
            try:
                st.write('one hot encoder')
                #x = OneHotEncoder(categories='auto',drop=None,handle_unknown='ignore').fit_transform(x[i])
                new_one_hot = pd.get_dummies(x[i], prefix=i)
                x = x.join(new_one_hot)
                x = x.drop([i], axis=1)
            except Exception:
                if df_l < 15:
                    st.write('label encoder')
                    le = LabelEncoder()
                    x[i] = le.fit_transform(x[i])
                    # st.write(oh_e)
                    # pass
        elif df_l < 15:
            st.write('label encoder')
            le = LabelEncoder()
            x[i] = le.fit_transform(x[i])
            # oh_e.fit_transform(x)
            # print(oh_e.categories_)
            # oh_e.transform(x)
            # print(x[i])
        else:
            st.write('can not apply encode')
            numerical_feature.append(i)
        # st.write('encoding',i,x,df_l)
            # else:
            # pass
    st.write(x)
    return x, numerical_feature


# """
# funtion preprocessing(x) is used for the pre-processing task on the data
# """
def preprocessing(x):
    #"""droping of duplicates"""
    st.write('Entered PreProcessing')
    #st.write('test -- 1')
    # st.write(x)
    x = x.drop_duplicates(keep='first')
    print('printing duplicate')
    #x.drop('datetime', axis=1)
    print(x.info())
    st.write(x.info())
    # st.write(x.describe())
    #st.write('test -- 2')
    # st.write(x)

    #"""handling of missing values"""
    print('----handling of missing values-----')
    print(x.isna().sum())
    rows_cnt = x.index[-1]
    rows_cnt += 1
    print(rows_cnt)
    col = list(x.columns)
    # col.remove('age')
    print(col)
    #st.write('test -- 3')
    # st.write(x)

    from pandas.api.types import is_datetime64_any_dtype as is_datetime
    #x[[column for column in x.columns if is_datetime(x[column])]]

    try:
        for i in col:
            # st.write(i)
            try:
                print(i, x[i].dtype)
                if x[i].dtype != object:
                    print('not datetime')
                    mis_per = x[i].isna().sum()/rows_cnt*100
                    print(mis_per)
                    if mis_per > 60:
                        #""" if missing values is greater than 60 %, then delete/drop it """
                        x.dropna(inplace=True)
                        break
                    # """ if missing values is not greater than 60 %, then do imputations \n
                    # previous values\n
                    # mean/median/mode, MICE, Linear Interpolation,\n
                    # Last Observation Carried Forward (LOCF) & Next Observation Carried Backward (NOCB)\n
                    # Time-Series Specific Methods, etc """
                    if not is_datetime(x[i]):
                        # imputing of mean value
                        x = x.fillna(x[i].mean())
                        #st.write('entered try if')
                        # st.write(i)
                else:
                    dt = i
                    print(dt)
                    print('entered else')
                    x[i] = pd.to_datetime(x[i])
                    #x = x.set_index(i)
                    print(x[i].dtype)
                    #st.write('entered try else')
            except Exception:
                continue
            #st.write('test -- 4')
            # st.write(x)
    except Exception:
        print('bug, somthing went wrong')

    print(x.info())
    print(x.isna().sum())
    # st.write(dt)
    #x = label_encode(x)
    # print(x['datetime'])
    #x[dt] = pd.to_datetime(x[dt])
    # print(x[dt])
    # cgn.is_gaussian(x)
    # st.write(x)
    #st.write('test -- 5')
    # st.write(x)

    return x


def shap_explain():
    import xgboost
    import shap
    import streamlit.components.v1 as components

    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
    fig2 = plt.figure()
    X, y = shap.datasets.boston()
    model = xgboost.XGBRegressor().fit(X, y)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    fig3 = shap.plots.waterfall(shap_values[0])
    plt.savefig('shap.jpg')
    st_shap(fig3)
    # st.pyplot(fig2)


def outlier(nw_dat, features):
    print('--------- outlier zone -----------')
    ndat = pd.DataFrame()
    with st.expander("outlier-plot"):
        st.write('outlier-Plot')
        with st.container():
            for c in features:
                if len(set(nw_dat[c])) > 3:
                    st.write(len(set(c)))
                    oulier_fig = plt.figure()
                    plt.title(c)
                    plt.boxplot(nw_dat[c])
                    st.pyplot(oulier_fig)
                    Q1 = nw_dat[c].quantile(0.25)
                    Q3 = nw_dat[c].quantile(0.75)
                    print(Q1, Q3)
                    IQR = Q3-Q1
                    print(IQR)
                    lowLim = Q1-1.5*IQR
                    uplim = Q3+1.5*IQR
                    print(lowLim, uplim)
                    ndat = nw_dat[(nw_dat[c] > lowLim) | (nw_dat[c] < uplim)]
                else:
                    ndat = nw_dat.copy()
    print("-----------------------------------------------------------")
    print(ndat)
    st.write("-----------ndat------------------of outlier-------------")
    st.write(ndat)
    return ndat


def predict(flag, nw_dat, features, target, type):
    # sourcery skip: avoid-builtin-shadow
    df2 = pd.DataFrame()
    res = pd.DataFrame(columns=['duration', 'rmse', 'r2', 'accuracy'])
    if flag:
        print('Flag is True here')
        print('for the 1st mont data is collected')

        #time_stamp = np.array(pd.to_numeric(list(nw_dat.index)).astype('Int32'))
        split_time = int(len(nw_dat.index)*60/100)
        test_split_time = int(len(nw_dat.index)*80/100)
        st.write('split time: ', split_time)
        train = nw_dat[:split_time]
        val = nw_dat[split_time:test_split_time]
        test = nw_dat[test_split_time:]
        # st.write('train:',train)

        #"""whole time line is used"""
        str = 'whole time line is used'
        print(str)
        X_t, X_v, y_t, y_v = train[features], val[features], train[target], val[target]
        df, model, preds, rmse, r2, acc = modeler(
            type, target, X_t, X_v, y_t, y_v)
        print('\n the model used is : ', model,
              '\n the rmse for the model is : ', rmse, '\n the r2 score is : ', r2)
        # df2=df2.append(df)
        res = res.append({'duration': str, 'rmse': rmse,
                         'r2': r2, 'accuracy': acc}, ignore_index=True)

        #""" --------graph----------- """
        err_graph(df, flag, type)
        #"""--------- shap --------"""
        # shap_explain()
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            nw_dat[features], nw_dat[target], test_size=0.8, random_state=42)
        df, model, preds, rmse, r2, acc = modeler(
            type, target, X_train, X_val, y_train, y_val)
        res = res.append({'duration': 'all', 'rmse': rmse,
                         'r2': r2, 'accuracy': acc}, ignore_index=True)
        #""" --------graph----------- """
        # err_graph(df,flag,type)

    st.write('result of original dataset train and predicting:')
    st.table(res)

    #go_for_ci_cd = st.button('go for ci/cd')
    # if go_for_ci_cd :
    count = 0
    st.write('count:', count)
    with st.expander("apply ci/cd"):
        st.write('apply ci/cd')
        with st.container():
            # st.write(":smile:")

            #countl = st_autorefresh(interval=120000, limit=30, key="counter")
            # if countl>=0:
            # st.write('count:',countl)
            cicd.apply(df, target)

    return model, rmse, r2


def data_to_aggrid(ag_data):
    st.write(ag_data.info())
    from st_aggrid.shared import GridUpdateMode

    def ag_sel_plot():
        import plotly_express as px
        selected_rows = ag_data["selected_rows"]
        selected_rows = pd.DataFrame(selected_rows)

        if len(selected_rows) != 0:
            fig = px.bar(selected_rows, "rating", color="type")
            st.plotly_chart(fig)

    # data in AgGrid data table
    # ag_data=nw_dat
    # AgGrid(ag_data)
    gd = GridOptionsBuilder.from_dataframe(ag_data)
    gd.configure_pagination(enabled=True)
    gd.configure_side_bar()
    gd.configure_default_column(
        groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    #gd.configure_default_column(editable=True, groupable=True)
    gd.configure_selection(selection_mode="multiple", use_checkbox=True)
    gridoptions = gd.build()
    #AgGrid(ag_data, gridOptions=gridoptions, enable_enterprise_modules=True, update_mode=GridUpdateMode.SELECTION_CHANGED)
    AgGrid(ag_data, gridOptions=gridoptions, enable_enterprise_modules=True)
    #grid_table = AgGrid(df, gridOptions=gridoptions, update_mode=GridUpdateMode.SELECTION_CHANGED,height=500,allow_unsafe_jscode=True, theme='fresh')
    #sel_row = grid_table["selected_rows"]
    # st.write(sel_row)
    print('---- plot selected rows in aggrid ----')
    # ag_sel_plot()


#""" modeling / training and testing on data """
def learn(dat, target, type):
    st.write('target is: ', target)
    # st.table(dat)
    print('--------- entered learn ------------')
    # target feature will be given by the user
    features = list(dat.columns)
    st.write('---------------------------------------------')
    # st.write(dat[:20])
    flag = False
    for i in features:
        # print(i,dat[i].dtype)
        try:
            if is_datetime(dat[i]):
                print('here is datetime')
                flag = True
                indx = i
                features.remove(i)
                print('datetime removed from features')
                dat = dat.sort_values(by=[i])
                dat = dat.set_index(dat[i])
                #del dat[i]
                print('sorted by datetime \nindex set as datetime')
                # for j in range(len(dat[i])):
                # j=2
                # if dat[i][j].dtype == datetime.datetime:
                # if i == 'time':
                #print('here is datetime')
                # features.remove(i)
                #""" datetime.datetime """
        except Exception:
            continue

    #""" feature engineering """
    print('---------feature engineering---------')
    st.write('new data')
    nw_dat = dat[features]
    st.write('original data with updated formate')
    # st.write(nw_dat)

    data_to_aggrid(nw_dat)
    st.write(nw_dat)

    nw_dat, num_feature = label_encode(nw_dat)
    features = list(nw_dat.columns)

    nw_dat = outlier(nw_dat, features)

    nw_dat = fea(nw_dat)

    if flag:
        nw_dat = nw_dat.set_index(dat[indx])
        # dat[features]=fea(dat[features])
        del dat[indx]
    st.write(nw_dat)
    print(dat.head(10))
    # st.write(nw_dat[:20])
    #nw_dat = outlier(nw_dat,features)
    features.remove(target)

    print(features)
    print(target)

    #""" modeleing  """
    print('--------- modeleling ---------')

    print('--------- showing_graph ---------')
    show_graph(nw_dat, features, target)

    print('--------- auto_correlation ---------')
    fig2 = plt.figure()
    autocorrelation_plot(nw_dat[target])
    plt.acorr(nw_dat[target])
    st.pyplot(fig2)

    """ check guassian """

    col = list(nw_dat[num_feature])
    list_guss = cgn.is_gaussian(nw_dat[col])
    st.write(list_guss)
    tonorm = [i for i in list_guss if i != 0]
    nw_dat[tonorm] = norm.norm(nw_dat[tonorm])
    st.write(tonorm)
    st.write('data after normalize cross check--------')
    st.write(nw_dat)

    with st.expander("qq-plot"):
        st.write('qq-Plot')
        with st.container():
            for i in tonorm:
                qq_fig = plt.figure()
                stats.probplot(nw_dat[i], dist="norm", plot=plt)
                plt.title("Normal Q-Q plot of : "+i)
                #qq_fig = pplot(nw_dat,x=nw_dat.index, y=nw_dat[i] , kind='qq')
                st.pyplot(qq_fig)
                #hist_fig = plt.figure()
                # plt.hist(nw_dat[i],20)
                # st.pyplot(hist_fig)
    list_guss_trans = cgn.is_gaussian(nw_dat[tonorm])
    st.write(list_guss_trans)

    if type == 3:
        """--------- time - series train --------"""
        #import time_series as ts
        #series=np.array(pd.to_numeric(dat['Monthly Mean Total Sunspot Number']))
        #time=np.array(pd.to_numeric(dat['Unnamed: 0']).astype('Int32'))
        # print('converted')
        #plt.figure(figsize=(10, 6))
        #ts.plot_series(time, series)
    # else:
    print('--------- predict ---------')
    model, rmse, r2 = predict(flag, nw_dat, features, target, type)

    # print(y_val)
    # print(pred_val)
    # print(df2.info())
    # print(nw_dat.head(5))
    return dat, model, rmse, r2

# """
# function data(x) to read dataset/n
#    where/n
#        'x' : filename
# """

# @st.cache(ttl=7200)
# @st.experimental_memo(suppress_st_warning=True)


def data(x, target, type):
    with st.spinner('Processing... \n please wait..'):
        #dat= pd.read_xml(x)
        # st.write(x)
        st.write('type:', type)
        print('Only CSV files are Allowed and Use Only Numeric and Categorical datasets')

        # try:
        if isinstance(x, pd.DataFrame):
            #st.write('enter if')
            dat = x
            print(dat.info())
            print(dat.tail(10))
            # AgGrid(dat)

        elif x.find('.csv') != -1:
            dat = pd.read_csv(x)
            print(dat.info())
            print(dat.tail(10))
            # AgGrid(dat)
        elif data_file:
            file_details = {"Filename": data_file.name, "FileType": data_file.type,
                            "FileSize (in kb)": round(data_file.size, 3)/1024}
            st.write(file_details)
            dat = pd.read_csv(data_file)
            #target = st.selectbox("select target",dat.columns)
            # AgGrid(dat)

        else:
            st.write('not loading data!!!!! \n use proper file type.')

        # if st.checkbox('Show Raw Data'):
            #process = True
            #st.subheader('Raw Data of :  ')
        st.write('original uploaded file')
        # st.write(dat)
        # AgGrid(dat)
        data_to_aggrid(dat)

        if type == 4:
            import pipe_try as pt
            pt.load(dat)
        elif type != 3:
            dat = preprocessing(dat)
            # st.write(dat)
            print(dat.info())
            # print(dat.columns)
            dat, model, rmse, r2_score = learn(dat, target, type)
            print('\n the model used is : ', model, '\n the rmse for the model is : ',
                  rmse, '\n the r2 score is : ', r2_score)
        else:
            """--------- time - series train --------"""
            import time_series as ts
            # @st.cache(ttl=7200)
            ts.data(x, target)
        print('done...all..:)')
        # print(nw_dat.head(5))
    st.success('hurray!! Sucessfully executed  \n;)  ')


# """
# calling the function data(filename)
# """

# if count==0:
selected_option = st.sidebar.selectbox(
    "Choose Dataset",
    ("None", "all_data_bike2", "house_price_data", "cyclonePreheater","inverter_data",
     "KAG_conversion_data2", "Sunspot", "out_clean", "winequality-white2", "other", "url", "g_sheet")
)
# selected_option = st.sidebar.selectbox(
#    "Choose Dataset",
#    ("None","regression","classification", "time series")
# )

with st.sidebar:
    st_m_type = st.selectbox(
        "select type", ("None", "regression", "classification", "time series", "pipe"))
    #st.write('* type: ',0,1,2,'regression')
    #st.write('* type: ',0,'= not a time data with normal ML')
    #st.write(1,',',2,'= time data with normal ML')
    #st.write(3,'= time_series')
    st.write(
        '** Use pipe only for kc_house_data.csv (available on kaggle), due to its testing phase')
    if st_m_type == 'None':
        st_type = 0
    elif st_m_type == 'regression':
        st_type = 1
    elif st_m_type == 'classification':
        st_type = 2
    elif st_m_type == 'time series':
        st_type = 3
    elif st_m_type == 'pipe':
        st_type = 4
    else:
        st.write('selct a model type')

    display = st.button('display')

    if apply_cicd := st.button('Apply CI'):
        st.write(":smile:")

    st.write('choose how you want to enter a data as a google sheet or just upload')
    data_file_up = st.selectbox(
        "how to upload a data", ("Drag or browse file", "google sheet"))

    if data_file_up == "Drag or browse file":
        # with st.sidebar:
        data_file = st.file_uploader("Upload CSV", type=['csv'])
        st.write(data_file)
        if data_file:
            df = pd.read_csv(data_file)
            target = st.selectbox("select target", df.columns)
            #data(df, target, type)
            #import time_series as ts
            # @st.cache(ttl=7200)
            #ts.data(df, target)

    elif data_file_up == "google sheet":
        #public_gsheets_url = "https://docs.google.com/spreadsheets/d/1lh9YUjvYfRIO7o88wgA3fpwxTlWrMKe36S5TFRwzbdI/edit?usp=sharing"

        # Create a connection object.
        conn = connect()
        with st.sidebar:
            public_gsheets_url = st.text_input('The google sheet URL link')
            st.write('only press enter with keyboard')
            st.write('DO NOT HIT PROCESS BEFORE SELECTING TARGET')
        #connected = http.client.HTTPConnection(public_gsheets_url)
        if public_gsheets_url:

            # Perform SQL query on the Google Sheet.
            # Uses st.cache to only rerun when the query changes or after 10 min = 600 ttl.
            # @st.experimental_memo(ttl=300)
            # @st.cache(ttl=75)
            st.write(public_gsheets_url)

            def run_query(query):
                rows = conn.execute(query, headers=1)  # , headers=1
                rows = rows.fetchall()  # st.write('runned sucess')
                return rows

            #sheet_url = st.secrets["public_gsheets_url"]
            sheet_url = public_gsheets_url
            rows = run_query(f'SELECT * FROM "{sheet_url}"')
            st.write('runned sucess')
            #st.write('rows are:',rows)
            gs_df = pd.DataFrame(rows)

            # Print results.
            # for row in rows:
            #st.write(f"{row.Name} has {row.age} age")
            if gs_df.shape[0] > 30:
                # with st.spinner('Loading your data from google sheets... \n please wait for a bit'):
                df = gs_df
                data_file = 'google sheet'
                target = st.selectbox("select target", df.columns)

            else:
                st.write(gs_df)
                #target = st.selectbox("select target",gs_df.columns)
                # st.write(target)
        else:
            st.write('enter the url, then again click display button')

        # with st.sidebar:
    # st.write(type(df))
    #data_dict = df.to_dict()
    # display
    # st.write(type(data_dict))
    #df = pd.DataFrame.from_dict(data_dict)
    # st.write(type(df))
    process = st.button("Process")

if process and data_file is not None:
    # st.dataframe(df)
    data(df, target, st_type)
    # data(data_file,'',st_type)
    # process=False
st.write(process)
# except Exception:
#st.write('either select file or upload a file....')
if display:
    if selected_option == 'None':
        st.write('NO DataSet selected')
        st.write(st_type)
        col1, col2, col3 = st.columns(3)
        #a,b,c = st.columns(3)
        st.write(type(col1))
        with col1:
            st.header("A cat")
            st.image("https://static.streamlit.io/examples/cat.jpg")

        with col2:
            st.header("A dog")
            st.image("https://static.streamlit.io/examples/dog.jpg")

        with col3:
            st.header("An owl")
            st.image("https://static.streamlit.io/examples/owl.jpg")

    elif selected_option == 'all_data_bike2':
        """ data with datetime """
        st.write('type 1 and type 3 compatible only')
        if st_type in [1]:
            data('all_data_bike2.csv', 'count', st_type)
        else:
            st.write('select appropriate type')

    elif selected_option == 'upload':
        data_file = st.file_uploader("Upload CSV", type=['csv'])
        file_details = {"Filename": data_file.name, "FileType": data_file.type,
                        "FileSize (in kb)": round(data_file.size, 3)/1024}
        st.write(file_details)
        df = pd.read_csv(data_file)
        target = st.selectbox("select target", df.columns)
        # if selected_option in df.columns:
        process = st.button("Process")

        if process and data_file is not None:
            st.dataframe(df)
            data(df, target, st_type)

    elif selected_option == 'house_price_data':
        st.write('type 1,2 and type 3 compatible only')
        """ data with datetime """
        if st_type in [1, 3]:
            # data('cyclonePreheater.csv','Cyclone_Gas_Outlet_Temp',2)
            data('data.csv', 'price', st_type)
        else:
            st.write('select appropriate type')

    elif selected_option == 'inverter_data':
        st.write('type 1,2 and type 3 compatible only')
        """ data with datetime """
        if st_type in [1, 3]:
            # data('cyclonePreheater.csv','Cyclone_Gas_Outlet_Temp',2)
            data('inverter_data3.csv', '1104500527', st_type)
        else:
            st.write('select appropriate type')
    elif selected_option == 'cyclonePreheater':
        st.write('type 1,2 and type 3 compatible only')
        """ data with datetime """
        if st_type in [1, 3]:
            # data('cyclonePreheater.csv','Cyclone_Gas_Outlet_Temp',2)
            data('cyclonePreheater7.csv', 'Cyclone_Gas_Outlet_Temp', st_type)
        else:
            st.write('select appropriate type')
    elif selected_option == 'KAG_conversion_data2':
        """ data without datetime """
        st.write('type 0 compatible only')
        if st_type in [1, 2]:
            data('KAG_conversion_data2.csv', 'conv_rate', st_type)
        else:
            st.write('select appropriate type')
    elif selected_option == 'Sunspot':
        """ data with datetime """
        st.write('type 1 and type 3 compatible only')
        if st_type in [1, 3]:
            data('Sunspots.csv', 'Monthly Mean Total Sunspot Number', st_type)
        else:
            st.write('select appropriate type')
    elif selected_option == 'out_clean':
        """ data with datetime """
        st.write('type 1 and type 3 compatible only')
        if st_type in [1, 3]:
            data('Out_clean.csv',
                 'Total_Generation_kwh', st_type)
        else:
            st.write('select appropriate type')
    elif selected_option == 'winequality-white2':
        """ data without datetime """
        st.write('type 0 compatible only')
        if st_type in [1, 2]:
            data('winequality-white2.csv', 'quality', st_type)
        else:
            st.write('select appropriate type')
    elif selected_option == 'other':
        """ other data's """
        data('climate.xml')
    elif selected_option == 'url':
        """data from url"""
        url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o_exog.csv'
        data(url, 'y', 4)
    elif selected_option == 'g_sheet':
        st.write('option entered for g-sheet')

        #public_gsheets_url = "https://docs.google.com/spreadsheets/d/1lh9YUjvYfRIO7o88wgA3fpwxTlWrMKe36S5TFRwzbdI/edit?usp=sharing"

        # Create a connection object.
        conn = connect()
        with st.sidebar:
            public_gsheets_url = st.text_input('The URL link')
        #connected = http.client.HTTPConnection(public_gsheets_url)
        if public_gsheets_url != '':
            # Perform SQL query on the Google Sheet.
            # Uses st.cache to only rerun when the query changes or after 10 min = 600 ttl.
            # @st.experimental_memo(ttl=300)
            # @st.cache(ttl=75)
            def run_query(query):
                rows = conn.execute(query, headers=1)
                rows = rows.fetchall()
                #st.write('runned sucess')
                return rows

            #sheet_url = st.secrets["public_gsheets_url"]
            sheet_url = public_gsheets_url
            rows = run_query(f'SELECT * FROM "{sheet_url}"')
            st.write('runned sucess')
            #st.write('rows are:',rows)
            gs_df = pd.DataFrame(rows)
            st.write(gs_df)

            # Print results.
            # for row in rows:
            #st.write(f"{row.Name} has {row.age} age")
        else:
            st.write('enter the url, then again click display button')

    # st.experimental_memo.clear()
    st.write('done')
    # st.rerun(120)
    # st.experimental_rerun()
    st.stop()


# --------------- google sheet link --------------------
# https://docs.google.com/spreadsheets/d/1aFyZa8pMgm9xBC5tRuZGiuEQp0v80nZ2kKrJP0cE9PY/edit#gid=0
# https://docs.google.com/spreadsheets/d/10e8l4tPOBUf4oGPM0TTub0_yiZQOruHy_tv2H0JiRCU/edit#gid=0


# generating git logs

# git --no-pager log > log.txt
# git log > log2.txt
# git log -p --all > log3.txt
# git log --pretty=format:"%ad - %an: %s" --after="2018-01-01" --until="2018-06-30" > git_log.txt
# git log --after="2020-3-20" --pretty=format:'Author : %an %nDate/Time :  %aD%nCommit : %s'   | paste  > log.txt
# npm install generate-changelog -g
    # changelog generate
