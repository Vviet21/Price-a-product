from flask import Flask, render_template, request, url_for, flash, redirect
import os
import selenium
from selenium import webdriver
import crawl
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# ...

app = Flask(__name__)



app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER = 'static/uploads/'
app.config['SECRET_KEY']=os.urandom(24).hex()
messages = [{'Name': 'Something',
             'k': '0',
             'p':'0'
             }]
Img_shopee,Img_lazada,Img_tiki,Img_sendo=[[]],[[]],[[]],[[]]

Link_shopee,Link_lazada,Link_tiki,Link_sendo=[[]],[[]],[[]],[[]]


mean_Price_s,mean_Price_l,mean_Price_t,mean_Price_se=[],[],[],[]

data = [[]]
@app.route('/p_def', methods=('GET', 'POST'))
def p_def():
    return render_template('predict_def.html')
@app.route('/p_result', methods=('GET', 'POST'))
def p_result():
    if request.method == 'POST':
        price = request.form['price']
        df = data[-1]
        model = LinearRegression()
        X = df.Price.values.reshape((-1,1))
        y = df.Sold.values
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.15)
        model.fit(x_train,y_train)

        x_x = model.predict([[int(price)]])
    return render_template('predict_result.html',x_x=x_x,price=price)
    


@app.route('/', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        Name = request.form['Name']
        K = 1 #request.form['k']
        P = 10 #request.form['p']
        
        if not Name:
            flash('Name is required!')
        # elif not K:
        #     flash('K is required!')
        # elif not P:
        #     flash('P is required!')    
        else:
            messages.append({'Name': Name, 'k': K,'p':P})
            return redirect(url_for('new_base'))
        # mean_Price_s.pop()
    return render_template('home.html')
@app.route('/history')
def index():
    return render_template('index.html', messages=messages)
@app.route('/new_base/')
def new_base():
    return render_template('new_base.html')



@app.route('/img_shopee/')
def show_shopee():
    return render_template('shopee.html',mean_Price_s=mean_Price_s[-1],messages=messages[-1],Img_shopee=Img_shopee,Link_shopee=Link_shopee)
@app.route('/img_lazada/')
def show_lazada():
    return render_template('lazada.html',mean_Price_l=mean_Price_l[-1],messages=messages[-1],Img_lazada=Img_lazada,Link_lazada=Link_lazada)
@app.route('/img_tiki/')
def show_tiki():
    return render_template('tiki.html',mean_Price_t=mean_Price_t[-1],messages=messages[-1],Img_tiki=Img_tiki,Link_tiki=Link_tiki)
@app.route('/img_sendo/')
def show_sendo():
    return render_template('sendo.html',mean_Price_se=mean_Price_se[-1],messages=messages[-1],Img_sendo=Img_sendo,Link_sendo=Link_sendo)







@app.route('/crawl_shopee/')
def SHOPEE_CRAWL():
    SHOPEE = crawl.Shopee(messages[-1]['Name'],float(messages[-1]['k']),float(messages[-1]['p']))
    SHOPEE.crawldata()
    SHOPEE.process()

    SHOPEE.displot()
    SHOPEE.hist_sold()
    SHOPEE.plot_price()
    SHOPEE.plot_sold()
    SHOPEE.scatter()
    SHOPEE.plot_kde_price()
    SHOPEE.plot_kde_sold()


    SHOPEE.plot_K()
    SHOPEE.plot_kde_K()
    SHOPEE.plot_price_K()
    SHOPEE.scatter_K()
    SHOPEE.box_k()
    SHOPEE.model_()
    
    data.append( SHOPEE.getDF())
    Img_shopee.append(SHOPEE.getImg())
    Link_shopee.append(SHOPEE.getLink())

    predict_price = SHOPEE.get_mean_Price()
    pred = SHOPEE.predict(predict_price)
    print("pred : ",pred)
    mean_Price_s.append(np.round(predict_price,2))
    return redirect(url_for('show_shopee'))
@app.route('/crawl_lazada/')
def LAZADA_CRAWL():
    LAZADA = crawl.Lazada(messages[-1]['Name'],float(messages[-1]['k']),float(messages[-1]['p']))
    LAZADA.l_crawldata()
    LAZADA.l_process()
    LAZADA.l_displot()
    LAZADA.l_hist_sold()
    LAZADA.l_plot_price()
    LAZADA.l_plot_sold()
    LAZADA.l_scatter()
    LAZADA.l_plot_kde_price()
    LAZADA.l_plot_kde_sold()

    LAZADA.l_plot_K()
    LAZADA.l_plot_kde_K()
    LAZADA.l_plot_price_K()
    LAZADA.l_scatter_K()
    LAZADA.l_box_k()
    Img_lazada.append(LAZADA.l_getImg())
    Link_lazada.append(LAZADA.l_getLink())

    predict_price = LAZADA.l_get_mean_Price()
    mean_Price_l.append(np.round(predict_price,2))

    return redirect(url_for('show_lazada'))
@app.route('/crawl_tiki/')
def TIKI_CRAWL():
    TIKI = crawl.Tiki(messages[-1]['Name'],float(messages[-1]['k']),float(messages[-1]['p']))
    TIKI.t_crawldata()
    TIKI.t_process()

    TIKI.t_displot()
    TIKI.t_hist_sold()
    TIKI.t_plot_price()
    TIKI.t_plot_sold()
    TIKI.t_scatter()
    TIKI.t_plot_kde_price()
    TIKI.t_plot_kde_sold()

    TIKI.t_plot_K()
    TIKI.t_plot_kde_K()
    TIKI.t_plot_price_K()
    TIKI.t_scatter_K()
    TIKI.t_box_k()
    Img_tiki.append(TIKI.t_getImg())
    Link_tiki.append(TIKI.t_getLink())

    predict_price = TIKI.t_get_mean_Price()
    mean_Price_t.append(np.round(predict_price,2))
    return redirect(url_for('show_tiki'))
@app.route('/crawl_sendo/')
def SENDO_CRAWL():
    SENDO = crawl.Sendo(messages[-1]['Name'],float(messages[-1]['k']),float(messages[-1]['p']))
    SENDO.s_crawldata()
    SENDO.s_process()

    SENDO.s_displot()
    SENDO.s_hist_sold()
    SENDO.s_plot_price()
    SENDO.s_plot_sold()
    SENDO.s_scatter()
    SENDO.s_plot_kde_price()
    SENDO.s_plot_kde_sold()

    SENDO.s_plot_K()
    SENDO.s_plot_kde_K()
    SENDO.s_plot_price_K()
    SENDO.s_scatter_K()
    SENDO.s_box_k()
    Img_sendo.append(SENDO.s_getImg())
    Link_sendo.append(SENDO.s_getLink())

    predict_price = SENDO.s_get_mean_Price()
    mean_Price_se.append(np.round(predict_price,2))
    return redirect(url_for('show_sendo'))