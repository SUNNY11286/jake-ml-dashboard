# MLD_orgi.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from math import sqrt
from gsheetsdb import connect
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.plotting import autocorrelation_plot
import streamlit as st
from PIL import Image
import check_guss_norm as cgn
import normalize as norm
import scipy.stats as stats
import cicd

class MLDashboard:
    def __init__(self):
        self.image = Image.open('MLDicon.jpg')
        self.setup_page()
        self.data_file = None
        self.df = None
        self.target = None
        self.model_type = None

    def setup_page(self):
        st.set_page_config(
            page_title="ML-dashboard",
            page_icon=self.image,
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/SUNNY11286/project_MLD_help',
                'Report a bug': "https://github.com/SUNNY11286/project_MLD_help",
                'About': "# This is part of project ML-D. \n This is an *testing version* web page!"
            }
        )
        st.title('JAKE \tML-Dashboard')
        st.write('WARNING \tit is under development so might be unstable.')
        print('ML Dashboard Begins')
        st.write('For any Issues and queries you can go the Get Help or Report section in Menu')

    def feature_engineering(self, x):
        col = list(x.columns)
        scaler = RobustScaler()
        st.write(x)
        robust_df = scaler.fit_transform(x)
        return pd.DataFrame(robust_df, columns=col)

    def modeler(self, model_type, target, X_t, X_v, y_t, y_v):
        df = pd.DataFrame()
        model = None
        preds = []
        rmse = r2 = acc = 0.0

        if model_type == 1:
            model = RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=2)
            model.fit(X_t, y_t)
            preds = model.predict(X_v)
            rmse = sqrt(mean_squared_error(y_v, preds))
            r2 = r2_score(y_v, preds)
            mape = mean_absolute_percentage_error(preds, y_v)
            acc = round(100 - mape.mean(), 2)
            df = pd.DataFrame(y_v)
            df['preds'] = list(preds)
            df['error'] = list(abs(df['preds'] - df[target]))
            print(df['preds'].count())

        elif model_type == 2:
            st.write('under development due to few errors / bugs')

        return df, model, preds, rmse, r2, acc

    def show_graph(self, nw_dat, features, target):
        with st.expander("pairplot"):
            st.write('Pair-Plot')
            pair_plot_fig = sns.pairplot(nw_dat, corner=True, hue=target)
            st.pyplot(pair_plot_fig)

        with st.expander("individual"):
            st.write('individual plots')
            for col in features:
                fig = plt.figure()
                plt.plot(nw_dat[col])
                plt.xticks(rotation=90)
                plt.xlabel('date')
                plt.ylabel(col)
                plt.title(f'no. of {col} against date/time')
                st.pyplot(fig)

        fig = plt.figure()
        plt.plot(nw_dat[target])
        plt.xticks(rotation=90)
        plt.xlabel('date')
        plt.ylabel(target)
        plt.title(f'no. of {target} against date/time')
        st.pyplot(fig)

    def err_graph(self, df, flag, model_type):
        print('------ enter error graph printing ------')

        erfig2 = plt.figure()
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
        split_time = int(len(df.index) * 80 / 100)
        st.write('error split time: ', split_time)
        train = df[:split_time]
        test = df[split_time:]

        def err_model(model_type, df):
            erres = pd.DataFrame(columns=['duration', 'rmse', 'r2'])
            if flag:
                str = 'whole time line is used'
                X_t, X_v, y_t, y_v = train[err_features], test[err_features], train[err_target], test[err_target]
                errdf, model, preds, rmse, r2, acc = self.modeler(
                    model_type, err_target, X_t, X_v, y_t, y_v)
                print('\n the model used is : ', model,
                      '\n the rmse for the model is : ', rmse, '\n the r2 score is : ', r2)
                erres = erres.append(
                    {'duration': str, 'rmse': rmse, 'r2': r2, 'accuracy': acc}, ignore_index=True)

                err_fig2 = plt.figure()
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
                errdf, model, preds, rmse, r2, acc = self.modeler(
                    err_target, X_train, X_test, y_train, y_test)
                erres = erres.append(
                    {'duration': 'all', 'rmse': rmse, 'r2': r2, 'accuracy': acc}, ignore_index=True)

            st.write('result of train on error and predicting error')
            st.table(erres)

        err_model(model_type, df)

    def label_encode(self, x):
        numerical_feature = []
        for i in x.columns:
            st.write(i)
            df_l = len(x[i].unique())
            st.write(df_l)
            if df_l < 7:
                try:
                    st.write('one hot encoder')
                    new_one_hot = pd.get_dummies(x[i], prefix=i)
                    x = x.join(new_one_hot)
                    x = x.drop([i], axis=1)
                except Exception:
                    if df_l < 15:
                        st.write('label encoder')
                        le = LabelEncoder()
                        x[i] = le.fit_transform(x[i])
            elif df_l < 15:
                st.write('label encoder')
                le = LabelEncoder()
                x[i] = le.fit_transform(x[i])
            else:
                st.write('can not apply encode')
                numerical_feature.append(i)
        st.write(x)
        return x, numerical_feature

    def preprocessing(self, x):
        st.write('Entered PreProcessing')
        x = x.drop_duplicates(keep='first')
        print('printing duplicate')
        print(x.info())
        st.write(x.info())

        print('----handling of missing values-----')
        print(x.isna().sum())
        rows_cnt = x.index[-1] + 1
        col = list(x.columns)

        try:
            for i in col:
                try:
                    print(i, x[i].dtype)
                    if x[i].dtype != object:
                        print('not datetime')
                        mis_per = x[i].isna().sum() / rows_cnt * 100
                        print(mis_per)
                        if mis_per > 60:
                            x.dropna(inplace=True)
                            break
                        if not is_datetime(x[i]):
                            x = x.fillna(x[i].mean())
                    else:
                        dt = i
                        print(dt)
                        print('entered else')
                        x[i] = pd.to_datetime(x[i])
                        print(x[i].dtype)
                except Exception:
                    continue
        except Exception:
            print('bug, something went wrong')

        print(x.info())
        print(x.isna().sum())
        return x

    def shap_explain(self):
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

    def outlier(self, nw_dat, features):
        print('--------- outlier zone -----------')
        ndat = pd.DataFrame()
        with st.expander("outlier-plot"):
            st.write('outlier-Plot')
            for c in features:
                if len(set(nw_dat[c])) > 3:
                    st.write(len(set(c)))
                    oulier_fig = plt.figure()
                    plt.title(c)
                    plt.boxplot(nw_dat[c])
                    st.pyplot(oulier_fig)
                    Q1 = nw_dat[c].quantile(0.25)
                    Q3 = nw_dat[c].quantile(0.75)
                    IQR = Q3 - Q1
                    lowLim = Q1 - 1.5 * IQR
                    uplim = Q3 + 1.5 * IQR
                    ndat = nw_dat[(nw_dat[c] > lowLim) | (nw_dat[c] < uplim)]
                else:
                    ndat = nw_dat.copy()
        print("-----------------------------------------------------------")
        print(ndat)
        st.write("-----------ndat------------------of outlier-------------")
        st.write(ndat)
        return ndat

    def predict(self, flag, nw_dat, features, target, model_type):
        df2 = pd.DataFrame()
        res = pd.DataFrame(columns=['duration', 'rmse', 'r2', 'accuracy'])
        if flag:
            print('Flag is True here')
            split_time = int(len(nw_dat.index) * 60 / 100)
            test_split_time = int(len(nw_dat.index) * 80 / 100)
            st.write('split time: ', split_time)
            train = nw_dat[:split_time]
            val = nw_dat[split_time:test_split_time]
            test = nw_dat[test_split_time:]

            str = 'whole time line is used'
            print(str)
            X_t, X_v, y_t, y_v = train[features], val[features], train[target], val[target]
            df, model, preds, rmse, r2, acc = self.modeler(
                model_type, target, X_t, X_v, y_t, y_v)
            print('\n the model used is : ', model,
                  '\n the rmse for the model is : ', rmse, '\n the r2 score is : ', r2)
            res = res.append({'duration': str, 'rmse': rmse,
                             'r2': r2, 'accuracy': acc}, ignore_index=True)

            self.err_graph(df, flag, model_type)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                nw_dat[features], nw_dat[target], test_size=0.8, random_state=42)
            df, model, preds, rmse, r2, acc = self.modeler(
                model_type, target, X_train, X_val, y_train, y_val)
            res = res.append({'duration': 'all', 'rmse': rmse,
                             'r2': r2, 'accuracy': acc}, ignore_index=True)

        st.write('result of original dataset train and predicting:')
        st.table(res)

        with st.expander("apply ci/cd"):
            st.write('apply ci/cd')
            cicd.apply(df, target)

        return model, rmse, r2

    def data_to_aggrid(self, ag_data):
        st.write(ag_data.info())
        from st_aggrid.shared import GridUpdateMode

        def ag_sel_plot():
            import plotly_express as px
            selected_rows = ag_data["selected_rows"]
            selected_rows = pd.DataFrame(selected_rows)

            if len(selected_rows) != 0:
                fig = px.bar(selected_rows, "rating", color="type")
                st.plotly_chart(fig)

        gd = GridOptionsBuilder.from_dataframe(ag_data)
        gd.configure_pagination(enabled=True)
        gd.configure_side_bar()
        gd.configure_default_column(
            groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gd.configure_selection(selection_mode="multiple", use_checkbox=True)
        gridoptions = gd.build()
        AgGrid(ag_data, gridOptions=gridoptions, enable_enterprise_modules=True)

    def learn(self, dat, target, model_type):
        st.write('target is: ', target)
        print('--------- entered learn ------------')
        features = list(dat.columns)
        st.write('---------------------------------------------')
        flag = False
        for i in features:
            try:
                if is_datetime(dat[i]):
                    print('here is datetime')
                    flag = True
                    indx = i
                    features.remove(i)
                    dat = dat.sort_values(by=[i])
                    dat = dat.set_index(dat[i])
            except Exception:
                continue

        print('---------feature engineering---------')
        st.write('new data')
        nw_dat = dat[features]
        st.write('original data with updated formate')

        self.data_to_aggrid(nw_dat)
        st.write(nw_dat)

        nw_dat, num_feature = self.label_encode(nw_dat)
        features = list(nw_dat.columns)

        nw_dat = self.outlier(nw_dat, features)

        nw_dat = self.feature_engineering(nw_dat)

        if flag:
            nw_dat = nw_dat.set_index(dat[indx])
            del dat[indx]
        st.write(nw_dat)
        print(dat.head(10))
        features.remove(target)

        print(features)
        print(target)

        print('--------- modeleling ---------')

        print('--------- showing_graph ---------')
        self.show_graph(nw_dat, features, target)

        print('--------- auto_correlation ---------')
        fig2 = plt.figure()
        autocorrelation_plot(nw_dat[target])
        plt.acorr(nw_dat[target])
        st.pyplot(fig2)

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
            for i in tonorm:
                qq_fig = plt.figure()
                stats.probplot(nw_dat[i], dist="norm", plot=plt)
                plt.title("Normal Q-Q plot of : " + i)
                st.pyplot(qq_fig)

        list_guss_trans = cgn.is_gaussian(nw_dat[tonorm])
        st.write(list_guss_trans)

        if model_type == 3:
            pass
        else:
            print('--------- predict ---------')
            model, rmse, r2 = self.predict(flag, nw_dat, features, target, model_type)

        return dat, model, rmse, r2

    def data(self, x, target, model_type):
        with st.spinner('Processing... \n please wait..'):
            st.write('type:', model_type)
            print('Only CSV files are Allowed and Use Only Numeric and Categorical datasets')

            if isinstance(x, pd.DataFrame):
                dat = x
                print(dat.info())
                print(dat.tail(10))

            elif x.find('.csv') != -1:
                dat = pd.read_csv(x)
                print(dat.info())
                print(dat.tail(10))

            elif self.data_file:
                file_details = {"Filename": self.data_file.name, "FileType": self.data_file.type,
                                "FileSize (in kb)": round(self.data_file.size, 3) / 1024}
                st.write(file_details)
                dat = pd.read_csv(self.data_file)

            else:
                st.write('not loading data!!!!! \n use proper file type.')

            st.write('original uploaded file')
            self.data_to_aggrid(dat)

            if model_type == 4:
                import pipe_try as pt
                pt.load(dat)
            elif model_type != 3:
                dat = self.preprocessing(dat)
                print(dat.info())
                dat, model, rmse, r2_score = self.learn(dat, target, model_type)
                print('\n the model used is : ', model, '\n the rmse for the model is : ',
                      rmse, '\n the r2 score is : ', r2_score)
            else:
                import time_series as ts
                ts.data(x, target)
            print('done...all..:)')
        st.success('hurray!! Sucessfully executed  \n;)  ')

    def run(self):
        selected_option = st.sidebar.selectbox(
            "Choose Dataset",
            ("None", "all_data_bike2", "house_price_data", "cyclonePreheater", "inverter_data",
             "KAG_conversion_data2", "Sunspot", "out_clean", "winequality-white2", "other", "url", "g_sheet")
        )

        with st.sidebar:
            st_m_type = st.selectbox(
                "select type", ("None", "regression", "classification", "time series", "pipe"))
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
                st.write('select a model type')

            display = st.button('display')

            if apply_cicd := st.button('Apply CI'):
                st.write(":smile:")

            st.write('choose how you want to enter a data as a google sheet or just upload')
            data_file_up = st.selectbox(
                "how to upload a data", ("Drag or browse file", "google sheet"))

            if data_file_up == "Drag or browse file":
                self.data_file = st.file_uploader("Upload CSV", type=['csv'])
                st.write(self.data_file)
                if self.data_file:
                    self.df = pd.read_csv(self.data_file)
                    self.target = st.selectbox("select target", self.df.columns)

            elif data_file_up == "google sheet":
                conn = connect()
                with st.sidebar:
                    public_gsheets_url = st.text_input('The google sheet URL link')
                    st.write('only press enter with keyboard')
                    st.write('DO NOT HIT PROCESS BEFORE SELECTING TARGET')
                if public_gsheets_url:
                    def run_query(query):
                        rows = conn.execute(query, headers=1)
                        rows = rows.fetchall()
                        return rows

                    sheet_url = public_gsheets_url
                    rows = run_query(f'SELECT * FROM "{sheet_url}"')
                    st.write('runned success')
                    gs_df = pd.DataFrame(rows)

                    if gs_df.shape[0] > 30:
                        self.df = gs_df
                        self.data_file = 'google sheet'
                        self.target = st.selectbox("select target", self.df.columns)
                    else:
                        st.write(gs_df)
                else:
                    st.write('enter the url, then again click display button')

            process = st.button("Process")

        if process and self.data_file is not None:
            self.data(self.df, self.target, st_type)

        if display:
            if selected_option == 'None':
                st.write('NO DataSet selected')
                st.write(st_type)
                col1, col2, col3 = st.columns(3)
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
                st.write('type 1 and type 3 compatible only')
                if st_type in [1]:
                    self.data('all_data_bike2.csv', 'count', st_type)
                else:
                    st.write('select appropriate type')

            elif selected_option == 'upload':
                self.data_file = st.file_uploader("Upload CSV", type=['csv'])
                file_details = {"Filename": self.data_file.name, "FileType": self.data_file.type,
                                "FileSize (in kb)": round(self.data_file.size, 3) / 1024}
                st.write(file_details)
                self.df = pd.read_csv(self.data_file)
                self.target = st.selectbox("select target", self.df.columns)
                process = st.button("Process")

                if process and self.data_file is not None:
                    st.dataframe(self.df)
                    self.data(self.df, self.target, st_type)

            elif selected_option == 'house_price_data':
                st.write('type 1,2 and type 3 compatible only')
                if st_type in [1, 3]:
                    self.data('data.csv', 'price', st_type)
                else:
                    st.write('select appropriate type')

            elif selected_option == 'inverter_data':
                st.write('type 1,2 and type 3 compatible only')
                if st_type in [1, 3]:
                    self.data('inverter_data3.csv', '1104500527', st_type)
                else:
                    st.write('select appropriate type')

            elif selected_option == 'cyclonePreheater':
                st.write('type 1,2 and type 3 compatible only')
                if st_type in [1, 3]:
                    self.data('cyclonePreheater7.csv', 'Cyclone_Gas_Outlet_Temp', st_type)
                else:
                    st.write('select appropriate type')

            elif selected_option == 'KAG_conversion_data2':
                st.write('type 0 compatible only')
                if st_type in [1, 2]:
                    self.data('KAG_conversion_data2.csv', 'conv_rate', st_type)
                else:
                    st.write('select appropriate type')

            elif selected_option == 'Sunspot':
                st.write('type 1 and type 3 compatible only')
                if st_type in [1, 3]:
                    self.data('Sunspots.csv', 'Monthly Mean Total Sunspot Number', st_type)
                else:
                    st.write('select appropriate type')

            elif selected_option == 'out_clean':
                st.write('type 1 and type 3 compatible only')
                if st_type in [1, 3]:
                    self.data('Out_clean.csv', 'Total_Generation_kwh', st_type)
                else:
                    st.write('select appropriate type')

            elif selected_option == 'winequality-white2':
                st.write('type 0 compatible only')
                if st_type in [1, 2]:
                    self.data('winequality-white2.csv', 'quality', st_type)
                else:
                    st.write('select appropriate type')

            elif selected_option == 'other':
                self.data('climate.xml')

            elif selected_option == 'url':
                url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o_exog.csv'
                self.data(url, 'y', 4)

            elif selected_option == 'g_sheet':
                st.write('option entered for g-sheet')
                conn = connect()
                with st.sidebar:
                    public_gsheets_url = st.text_input('The URL link')
                if public_gsheets_url != '':
                    def run_query(query):
                        rows = conn.execute(query, headers=1)
                        rows = rows.fetchall()
                        return rows

                    sheet_url = public_gsheets_url
                    rows = run_query(f'SELECT * FROM "{sheet_url}"')
                    st.write('runned success')
                    gs_df = pd.DataFrame(rows)
                    st.write(gs_df)
                else:
                    st.write('enter the url, then again click display button')

        st.write('done')
        st.stop()

if __name__ == "__main__":
    dashboard = MLDashboard()
    dashboard.run()