
import selenium
from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class Shopee:
    #init
    def __init__(self,name,k,p):
        self.name=name
        self.k=k
        self.p=p
        self.df = pd.DataFrame()
        self.model = 0

    def load_page(self,driver):
        step = 600
        for i in range(1,7):
            end = step*i
            cur = "window.scrollTo(0,"+str(end)+")"
            driver.execute_script(cur)
            time.sleep(1)

    #predict
    def model_(self):
        model = LinearRegression()
        X = self.df.Price.values.reshape((-1,1))
        y = self.df.Sold.values
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.15)
        model.fit(x_train,y_train)
        self.model = model


    def predict(self,x):
        pred = self.model.predict([[x]])
        return pred
        


    #get
    def get_mode_Price(self):
        return np.mean(self.df.Price)
    def getDF(self):
        df1 = self.df
        return df1
    def get_mean_Price(self):
        return np.mean(self.df.Price)
    def getSold(self):
        print(self.df.Sold)
    def getImg(self):
        return self.df.Img.head(5)
    def getLink(self):
        return self.df.Link.head(5)


    # draw chart
    #base
    def displot(self):
        sigma = self.df.Price.std()
        mu = self.df.Price.mean()
        count,bins, ignored = plt.hist(self.df.Price,50,density=True,label='Price')
        plt.title(label='Price')
        plt.legend()
        plt.plot(bins,1/(sigma * np.sqrt(2*np.pi))*
                np.exp(-((bins-mu)**2) / (2*sigma**2)),linewidth=2,color='r')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/shopee/base/'
        plt.savefig(path + 'hist_price.png',dpi=300)
        plt.close()

    def hist_sold(self):
        hist_sold = plt.hist(self.df.Sold.head(50),50,density=True,label='Sold')
        plt.title(label='Sold',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/shopee/base/'
        plt.savefig(path + 'hist_sold.png',dpi=300)
        plt.close()


    def plot_price(self):
        plot_price = self.df.Price.plot()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/shopee/base/'
        plt.savefig(path + 'plot_price.png',dpi=300)
        plt.close()


    def plot_sold(self):
        plot_sold = self.df.Sold.plot()
        plt.title(label='Sold',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/shopee/base/'
        plt.savefig(path + 'plot_sold.png',dpi=300)
        plt.close()


    def scatter(self):
        scatter=self.df.plot.scatter(x='Price',y='Sold',c='DarkBlue')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/shopee/base/'
        plt.savefig(path + 'scatter.png',dpi=300)
        plt.close()


    def plot_kde_price(self):
        plot_kde_price = self.df.Price.plot.kde()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/shopee/base/'
        plt.savefig(path + 'kde_price.png',dpi=300)
        plt.close()


    def plot_kde_sold(self):
        plot_kde_sold = self.df.Sold.plot.kde()
        plt.title(label='Sold',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/shopee/base/'
        plt.savefig(path + 'kde_sold.png',dpi=300)
        plt.close()



    #price
    def plot_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        test = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        sigma = self.df.Price.std()
        mu = self.df.Price.mean()
        count,bins, ignored = plt.hist(self.df.Price,50,density=True,label='base')
        plt.hist(test.Price,50,density=True,label='recommend')
        plt.title(label='{}<Price<{}'.format(x1_,x2_),y=-0.15)
        plt.legend()
        plt.plot(bins,1/(sigma * np.sqrt(2*np.pi))*
                np.exp(-((bins-mu)**2) / (2*sigma**2)),linewidth=2,color='r')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/shopee/price/'
        plt.savefig(path + 'hist_k.png',dpi=300)
        plt.close()


    def plot_kde_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        plot_kde_K = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        plot_kde_K.Price.plot.kde()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/shopee/price/'
        plt.savefig(path + 'kde_k.png',dpi=300)
        plt.close()


    def plot_price_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        plot_price_K = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        plot_price_K.Price.plot()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/shopee/price/'
        plt.savefig(path + 'plot_price_k.png',dpi=300)
        plt.close()


    def scatter_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        scatter_K = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        scatter_K.plot.scatter(x='Price',y='Sold',c='DarkBlue')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/shopee/price/'
        plt.savefig(path + 'scatter_k.png',dpi=300)
        plt.close()

    
    def box_k(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        box_k = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        box_k.boxplot(column=['Price'])
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/shopee/price/'
        plt.savefig(path + 'box_k.png',dpi=300)
        plt.close()



    #crawl and process
    def crawldata(self):
        driver = webdriver.Chrome()
        url = 'https://shopee.vn/search?keyword={}'.format(self.name)

        driver.set_window_size(1300,800)
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source,"html.parser")
        total_page = int(soup.find('span',class_='shopee-mini-page-controller__total').get_text())
        print('Total_page : ',total_page)
        if total_page >10:
            total_page=10
        a,c,d,b,e = [],[],[],[],[]
        base_url='https://shopee.vn'
        page=1
        for i in range(0,total_page):
            self.load_page(driver)
            # time.sleep(2)
            soup = BeautifulSoup(driver.page_source,"html.parser")
            for _ in soup.find_all('div',{'class':'col-xs-2-4 shopee-search-item-result__item'}):
                name = _.find('div',{'class':'ie3A+n bM+7UW Cve6sh'})
                if name != None:
                    name = name.get_text()
                    if self.name.lower() in name.lower():
                        img = _.find('img').get('src')
                        price = _.find('span',{'class':'ZEgDH9'}).get_text()
                        sold = _.find('div',{'class':'r6HknA uEPGHT'})
                        if sold!=None:
                            sold =sold.get_text()
                        link = base_url+_.find('a').get('href')
                        
                        a.append(name)
                        b.append(img)
                        c.append(price)
                        d.append(sold)
                        e.append(link)
                        
            print('page ',page,' done!')
            
            if page==total_page:
                
                break
            page+=1
            driver.find_element(By.CSS_SELECTOR,'.shopee-icon-button.shopee-icon-button--right').click()
        
        self.df = pd.DataFrame(list(zip(a,c,d,b,e)),columns=['Name','Price','Sold','Img','Link'])
        driver.close()
         
    def process(self):
        
        for i,row in self.df.iterrows():
            items = row['Sold']
            if(type(items)!=str):
                continue
            if items != None:
                self.df.at[i,'Sold']=items.split()[2]

        for i,row in self.df.iterrows():
            items = row['Price']
            if(type(items)!=str):
                continue
            if items != None:
                if '.' in items:
                    if len(items.split('.')) == 3:
                        tmp = items.split('.')[2] 
                        if 'đ' in tmp :
                            tmp = tmp[:-1]
                        self.df.at[i,'Price']=(float(items.split('.')[0])*1000000)+((float(items.split('.')[1]))*1000)+((float(tmp)))
                    else :
                        self.df.at[i,'Price']=(float(items.split('.')[0])*1000)+(float(items.split('.')[1]))

        for i,row in self.df.iterrows():
            items = row['Sold']
            if(type(items)!=str):
                continue
            if items != None:
                if ',' in items:
                    self.df.at[i,'Sold']=(float(items.split(',')[0])*1000)+(float(items.split(',')[1][0]))*100
        for i,row in self.df.iterrows():
            items = row['Sold']
            if(type(items)!=str):
                continue
            if items != None:
                if 'k' in items:
                    self.df.at[i,'Sold']=(float(items.split('k')[0])*1000)

        self.df.Sold = self.df.Sold.astype('float')
        self.df.Price= self.df.Price.astype('float')
        self.df.Sold = self.df.Sold.fillna(self.df.Sold.mean())
        self.df.Price= self.df.Price.fillna(self.df.Price.mean())


class Lazada(Shopee):
    def l_get_mean_Price(self):
        return np.mean(self.df.Price)
    def l_getImg(self):
        return self.df.Img.head(5)
    def l_getLink(self):
        return self.df.Link.head(5)
    
    def l_displot(self):
        sigma = self.df.Price.std()
        mu = self.df.Price.mean()
        count,bins, ignored = plt.hist(self.df.Price,50,density=True,label='Price')
        plt.title(label='Price')
        plt.legend()
        plt.plot(bins,1/(sigma * np.sqrt(2*np.pi))*
                np.exp(-((bins-mu)**2) / (2*sigma**2)),linewidth=2,color='r')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/lazada/base/'
        plt.savefig(path + 'hist_price.png',dpi=300)
        plt.close()

    def l_hist_sold(self):
        hist_sold = plt.hist(self.df.Sold.head(50),50,density=True,label='Sold')
        plt.title(label='Sold',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/lazada/base/'
        plt.savefig(path + 'hist_sold.png',dpi=300)
        plt.close()


    def l_plot_price(self):
        plot_price = self.df.Price.plot()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/lazada/base/'
        plt.savefig(path + 'plot_price.png',dpi=300)
        plt.close()


    def l_plot_sold(self):
        plot_sold = self.df.Sold.plot()
        plt.title(label='Sold',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/lazada/base/'
        plt.savefig(path + 'plot_sold.png',dpi=300)
        plt.close()


    def l_scatter(self):
        scatter=self.df.plot.scatter(x='Price',y='Sold',c='DarkBlue')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/lazada/base/'
        plt.savefig(path + 'scatter.png',dpi=300)
        plt.close()


    def l_plot_kde_price(self):
        plot_kde_price = self.df.Price.plot.kde()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/lazada/base/'
        plt.savefig(path + 'kde_price.png',dpi=300)
        plt.close()


    def l_plot_kde_sold(self):
        plot_kde_sold = self.df.Sold.plot.kde()
        plt.title(label='Sold',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/lazada/base/'
        plt.savefig(path + 'kde_sold.png',dpi=300)
        plt.close()



    #price
    def l_plot_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        test = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        sigma = self.df.Price.std()
        mu = self.df.Price.mean()
        count,bins, ignored = plt.hist(self.df.Price,50,density=True,label='base')
        plt.hist(test.Price,50,density=True,label='recommend')
        plt.title(label='{}<Price<{}'.format(x1_,x2_),y=-0.15)
        plt.legend()
        plt.plot(bins,1/(sigma * np.sqrt(2*np.pi))*
                np.exp(-((bins-mu)**2) / (2*sigma**2)),linewidth=2,color='r')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/lazada/price/'
        plt.savefig(path + 'hist_k.png',dpi=300)
        plt.close()


    def l_plot_kde_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        plot_kde_K = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        plot_kde_K.Price.plot.kde()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/lazada/price/'
        plt.savefig(path + 'kde_k.png',dpi=300)
        plt.close()


    def l_plot_price_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        plot_price_K = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        plot_price_K.Price.plot()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/lazada/price/'
        plt.savefig(path + 'plot_price_k.png',dpi=300)
        plt.close()


    def l_scatter_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        scatter_K = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        scatter_K.plot.scatter(x='Price',y='Sold',c='DarkBlue')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/lazada/price/'
        plt.savefig(path + 'scatter_k.png',dpi=300)
        plt.close()

    
    def l_box_k(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        box_k = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        box_k.boxplot(column=['Price'])
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/lazada/price/'
        plt.savefig(path + 'box_k.png',dpi=300)
        plt.close()

    
    

    def l_process(self):
        for i,row in self.df.iterrows():
            items = row['Sold']
            if(type(items)!=str):
                continue
            if items != None:
                self.df.at[i,'Sold']=items.split()[0]
                if '+' in self.df.at[i,'Sold']:
                    self.df.at[i,'Sold']=items.split()[0][:-1]
                if ',' in self.df.at[i,'Sold']:
                    self.df.at[i,'Sold']=(float(items.split(',')[0])*1000)+(float(items.split(',')[1][0]))*100
        for i,row in self.df.iterrows():
            items = row['Price']
            if(type(items)!=str):
                continue
            if items != None:
                if '-' in items :
                    self.df.at[i,'Price']=items.split('- ')[1]
                if self.df.at[i,'Price'][-1] =='₫':
                    self.df.at[i,'Price']=self.df.at[i,'Price'].split()[0]
                if '.' in self.df.at[i,'Price']:
                    if len(self.df.at[i,'Price'].split('.')) == 3:
                        tmp = self.df.at[i,'Price'].split('.')[2] 
                        if '₫' in tmp :
                            tmp = tmp[:-1]
                        self.df.at[i,'Price']=(float(self.df.at[i,'Price'].split('.')[0])*1000000)+((float(self.df.at[i,'Price'].split('.')[1]))*1000)+((float(tmp)))
                    else :
                        self.df.at[i,'Price']=(float(self.df.at[i,'Price'].split('.')[0])*1000)+(float(self.df.at[i,'Price'].split('.')[1]))
        self.df.Sold = self.df.Sold.astype('float')
        self.df.Price= self.df.Price.astype('float')

    def l_crawldata(self):
        driver = webdriver.Chrome()
        url = 'https://www.lazada.vn/catalog/?spm=a2o4n.home.search.1.19053bdcVqgG5l&q={}'.format(self.name)

        driver.set_window_size(1300,800)
        driver.get(url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source,'html.parser')
        total_page=soup.find_all('li',class_='ant-pagination-item')[-1].text
        total_page=int(total_page)
        print('Total_page : ',total_page)
        a,b,c,d,e = [],[],[],[],[]
        base_url='https:'
        page=1
        for i in range(0,3):
                self.load_page(driver)
                soup = BeautifulSoup(driver.page_source,"html.parser")
                for _ in soup.find_all('div',{'class':'Bm3ON'}):
                    name = _.find('div',class_='RfADt').get_text()
                    if self.name.lower() in name.lower():
                        img = _.find('img').get('src')
                        price = _.find('span',class_='ooOxS').get_text()
                        sold = _.find('span',class_='_1cEkb')
                        if sold!=None:
                            sold =sold.get_text()
                        link = base_url + _.find('a').get('href')

                        # location = _.find('span',class_='oa6ri').get_text()
                        a.append(name)
                        b.append(img)
                        c.append(price)
                        d.append(sold)
                        e.append(link)
                        # f.append(location)
                
                print('page ',page,' done!')
                page+=1
                
                if page == total_page:
                    break
                driver.find_element(By.CLASS_NAME,'ant-pagination-next').click()
        self.df = pd.DataFrame(list(zip(a,c,d,b,e)),columns=['Name','Price','Sold','Img','Link'])
        driver.close()




class Tiki(Shopee):
    def t_getImg(self):
        return self.df.Img.head(5)
    def t_getLink(self):
        return self.df.Link.head(5)
    def t_get_mean_Price(self):
        return np.mean(self.df.Price)

    def t_displot(self):
        sigma = self.df.Price.std()
        mu = self.df.Price.mean()
        count,bins, ignored = plt.hist(self.df.Price,50,density=True,label='Price')
        plt.title(label='Price')
        plt.legend()
        plt.plot(bins,1/(sigma * np.sqrt(2*np.pi))*
                np.exp(-((bins-mu)**2) / (2*sigma**2)),linewidth=2,color='r')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/tiki/base/'
        plt.savefig(path + 'hist_price.png',dpi=300)
        plt.close()

    def t_hist_sold(self):
        hist_sold = plt.hist(self.df.Sold.head(50),50,density=True,label='Sold')
        plt.title(label='Sold',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/tiki/base/'
        plt.savefig(path + 'hist_sold.png',dpi=300)
        plt.close()


    def t_plot_price(self):
        plot_price = self.df.Price.plot()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/tiki/base/'
        plt.savefig(path + 'plot_price.png',dpi=300)
        plt.close()


    def t_plot_sold(self):
        plot_sold = self.df.Sold.plot()
        plt.title(label='Sold',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/tiki/base/'
        plt.savefig(path + 'plot_sold.png',dpi=300)
        plt.close()


    def t_scatter(self):
        scatter=self.df.plot.scatter(x='Price',y='Sold',c='DarkBlue')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/tiki/base/'
        plt.savefig(path + 'scatter.png',dpi=300)
        plt.close()


    def t_plot_kde_price(self):
        plot_kde_price = self.df.Price.plot.kde()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/tiki/base/'
        plt.savefig(path + 'kde_price.png',dpi=300)
        plt.close()


    def t_plot_kde_sold(self):
        plot_kde_sold = self.df.Sold.plot.kde()
        plt.title(label='Sold',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/tiki/base/'
        plt.savefig(path + 'kde_sold.png',dpi=300)
        plt.close()



    #price
    def t_plot_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        test = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        sigma = self.df.Price.std()
        mu = self.df.Price.mean()
        count,bins, ignored = plt.hist(self.df.Price,50,density=True,label='base')
        plt.hist(test.Price,50,density=True,label='recommend')
        plt.title(label='{}<Price<{}'.format(x1_,x2_),y=-0.15)
        plt.legend()
        plt.plot(bins,1/(sigma * np.sqrt(2*np.pi))*
                np.exp(-((bins-mu)**2) / (2*sigma**2)),linewidth=2,color='r')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/tiki/price/'
        plt.savefig(path + 'hist_k.png',dpi=300)
        plt.close()


    def t_plot_kde_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        plot_kde_K = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        plot_kde_K.Price.plot.kde()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/tiki/price/'
        plt.savefig(path + 'kde_k.png',dpi=300)
        plt.close()


    def t_plot_price_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        plot_price_K = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        plot_price_K.Price.plot()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/tiki/price/'
        plt.savefig(path + 'plot_price_k.png',dpi=300)
        plt.close()


    def t_scatter_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        scatter_K = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        scatter_K.plot.scatter(x='Price',y='Sold',c='DarkBlue')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/tiki/price/'
        plt.savefig(path + 'scatter_k.png',dpi=300)
        plt.close()

    
    def t_box_k(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        box_k = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        box_k.boxplot(column=['Price'])
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/tiki/price/'
        plt.savefig(path + 'box_k.png',dpi=300)
        plt.close()


    
    def t_process(self):
        for i,row in self.df.iterrows():
            items = row['Price']
            if(type(items)!=str):
                continue
            if items != None:
                if '-' in items:
                    self.df.at[i,'Price'] = items.split('- ')[1]
                if self.df.at[i,'Price'][-1] =='₫':
                    self.df.at[i,'Price']=self.df.at[i,'Price'].split()[0]
                if '.' in self.df.at[i,'Price']:
                    if len(self.df.at[i,'Price'].split('.')) == 3:
                        tmp = self.df.at[i,'Price'].split('.')[2] 
                        if '₫' in tmp :
                            tmp = tmp[:-1]
                        self.df.at[i,'Price']=(float(self.df.at[i,'Price'].split('.')[0])*1000000)+((float(self.df.at[i,'Price'].split('.')[1]))*1000)+((float(tmp)))
                    else :
                        self.df.at[i,'Price']=(float(self.df.at[i,'Price'].split('.')[0])*1000)+(float(self.df.at[i,'Price'].split('.')[1]))
        for i,row in self.df.iterrows():
            items = row['Sold']
            if(type(items)!=str):
                continue
            if items != None:
                self.df.at[i,'Sold']=items.split()[2]
                if '+' in self.df.at[i,'Sold']:
                    self.df.at[i,'Sold']=items.split()[2][:-1]
        self.df.Sold = self.df.Sold.astype('float')
        self.df.Price= self.df.Price.astype('float')

    def t_crawldata(self):
        driver = webdriver.Chrome()
        url = 'https://tiki.vn/search?q={}'.format(self.name)
        driver.get(url)
        # time.sleep(2)
        soup = BeautifulSoup(driver.page_source,"html.parser")
        driver.set_window_size(1300,800)
        total_page = soup.find('span',class_='last').text
        total_page = int(total_page)
        a,b,c,d,e = [],[],[],[],[]
        base_url='https://tiki.vn'
        page=1
        for i in range(1,3+1):
            url_tiki = 'https://tiki.vn/search?q={}&page={}'.format(self.name,i)
            driver.get(url_tiki)
            self.load_page(driver)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source,"html.parser")
            for _ in soup.find_all('a',{'class':'product-item'}):
                name = _.find('div',class_='name').text
                if self.name.lower() in name.lower() :
                    price = _.find('div',class_='price-discount__price').text
                    
                    img = _.find('img')
                    if img != None:
                        img = img.get('src')
                    sold = _.find('div',class_='styles__StyledQtySold-sc-732h27-2 fCfYNm')
                    if sold!=None:
                        sold =sold.get_text()
                    link = base_url+_.get('href')
                    
                    a.append(name)
                    b.append(img)
                    c.append(price)
                    d.append(sold)
                    e.append(link)
                
            print('page ',page,' done!')
            page+=1
        self.df = pd.DataFrame(list(zip(a,c,d,e,b)),columns=['Name','Price','Sold','Link','Img'])
        driver.close()




class Sendo(Shopee):
    def s_getImg(self):
        return self.df.Img.head(5)
    def s_getLink(self):
        return self.df.Link.head(5)
    def s_get_mean_Price(self):
        return np.mean(self.df.Price)   

    #base
    def s_displot(self):
        sigma = self.df.Price.std()
        mu = self.df.Price.mean()
        count,bins, ignored = plt.hist(self.df.Price,50,density=True,label='Price')
        plt.title(label='Price')
        plt.legend()
        plt.plot(bins,1/(sigma * np.sqrt(2*np.pi))*
                np.exp(-((bins-mu)**2) / (2*sigma**2)),linewidth=2,color='r')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/sendo/base/'
        plt.savefig(path + 'hist_price.png',dpi=300)
        plt.close()

    def s_hist_sold(self):
        hist_sold = plt.hist(self.df.Sold.head(50),50,density=True,label='Sold')
        plt.title(label='Sold',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/sendo/base/'
        plt.savefig(path + 'hist_sold.png',dpi=300)
        plt.close()


    def s_plot_price(self):
        plot_price = self.df.Price.plot()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/sendo/base/'
        plt.savefig(path + 'plot_price.png',dpi=300)
        plt.close()


    def s_plot_sold(self):
        plot_sold = self.df.Sold.plot()
        plt.title(label='Sold',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/sendo/base/'
        plt.savefig(path + 'plot_sold.png',dpi=300)
        plt.close()


    def s_scatter(self):
        scatter=self.df.plot.scatter(x='Price',y='Sold',c='DarkBlue')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/sendo/base/'
        plt.savefig(path + 'scatter.png',dpi=300)
        plt.close()


    def s_plot_kde_price(self):
        plot_kde_price = self.df.Price.plot.kde()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/sendo/base/'
        plt.savefig(path + 'kde_price.png',dpi=300)
        plt.close()


    def s_plot_kde_sold(self):
        plot_kde_sold = self.df.Sold.plot.kde()
        plt.title(label='Sold',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/sendo/base/'
        plt.savefig(path + 'kde_sold.png',dpi=300)
        plt.close()



    #price
    def s_plot_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        test = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        sigma = self.df.Price.std()
        mu = self.df.Price.mean()
        count,bins, ignored = plt.hist(self.df.Price,50,density=True,label='base')
        plt.hist(test.Price,50,density=True,label='recommend')
        plt.title(label='{}<Price<{}'.format(x1_,x2_),y=-0.15)
        plt.legend()
        plt.plot(bins,1/(sigma * np.sqrt(2*np.pi))*
                np.exp(-((bins-mu)**2) / (2*sigma**2)),linewidth=2,color='r')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/sendo/price/'
        plt.savefig(path + 'hist_k.png',dpi=300)
        plt.close()


    def s_plot_kde_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        plot_kde_K = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        plot_kde_K.Price.plot.kde()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/sendo/price/'
        plt.savefig(path + 'kde_k.png',dpi=300)
        plt.close()


    def s_plot_price_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        plot_price_K = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        plot_price_K.Price.plot()
        plt.title(label='Price',y=-0.15)
        plt.legend()
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/sendo/price/'
        plt.savefig(path + 'plot_price_k.png',dpi=300)
        plt.close()


    def s_scatter_K(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        scatter_K = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        scatter_K.plot.scatter(x='Price',y='Sold',c='DarkBlue')
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/sendo/price/'
        plt.savefig(path + 'scatter_k.png',dpi=300)
        plt.close()

    
    def s_box_k(self):
        x1_= np.mean(self.df.Price)-np.std(self.df.Price) 
        x2_= np.mean(self.df.Price)+np.std(self.df.Price) 
        # print(' Giá từ ',x1_,' --- >',x2_)
        box_k = self.df.query('Price>={} and Price<={} '.format(x1_,x2_))
        box_k.boxplot(column=['Price'])
        path = 'C:/Users/a3785/OneDrive/Máy tính/ThucTap2023/TestApp/static/image/sendo/price/'
        plt.savefig(path + 'box_k.png',dpi=300)
        plt.close()




    def s_process(self):
        for i,row in self.df.iterrows():
            items = row['Price']
            if(type(items)!=str):
                continue
            if items != None:
                if '-' in items:
                    self.df.at[i,'Price']=items.split('- ')[1]
                if self.df.at[i,'Price'][-1] =='đ':
                    self.df.at[i,'Price']=self.df.at[i,'Price'][:-1]
                if '.' in self.df.at[i,'Price']:
                    if len(self.df.at[i,'Price'].split('.')) == 3:
                        tmp = self.df.at[i,'Price'].split('.')[2] 
                        if 'đ' in tmp :
                            tmp = tmp[:-1]
                        self.df.at[i,'Price']=(float(self.df.at[i,'Price'].split('.')[0])*1000000)+((float(self.df.at[i,'Price'].split('.')[1]))*1000)+((float(tmp)))
                    else :
                        self.df.at[i,'Price']=(float(self.df.at[i,'Price'].split('.')[0])*1000)+(float(self.df.at[i,'Price'].split('.')[1]))
        for i,row in self.df.iterrows():
            items = row['Sold']
            if(type(items)!=str):
                continue
            if items == '':
                self.df.at[i,'Sold'] = None
            if items != None and items != '':
                self.df.at[i,'Sold']=float(items.split()[2])
                
        self.df.Sold = self.df.Sold.astype('float')
        self.df.Price= self.df.Price.astype('float')


    def s_crawldata(self):
        driver = webdriver.Chrome()
        url = 'https://www.sendo.vn/tim-kiem?q={}'.format(self.name)

        driver.set_window_size(1300,800)
        driver.get(url)
        time.sleep(2)
        driver.execute_script('window.scrollTo(0, 5000)')
        time.sleep(2)
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
        time.sleep(2)
        for i in range(0,5):
            time.sleep(2)
            driver.find_element(By.CSS_SELECTOR,'.d7ed-BjtR6B.d7ed-H2lumk.d7ed-fdSIZS.d7ed-AwHm4T').click()
            time.sleep(2)
            driver.execute_script('window.scrollTo(0, 2500)')

            
        # driver.find_element(By.CSS_SELECTOR,'.d7ed-BjtR6B.d7ed-H2lumk.d7ed-fdSIZS.d7ed-AwHm4T').click()
        # driver.execute_script('window.scrollTo(0, 2500)')
        # time.sleep(2)
        # driver.find_element(By.CSS_SELECTOR,'.d7ed-BjtR6B.d7ed-H2lumk.d7ed-fdSIZS.d7ed-AwHm4T').click()
        # driver.execute_script('window.scrollTo(0, 2500)')
        # time.sleep(2)
        
        # for i in range(0,3):
        #     time.sleep(2)
        #     driver.find_element(By.CSS_SELECTOR,'.d7ed-BjtR6B.d7ed-H2lumk.d7ed-fdSIZS.d7ed-AwHm4T').click()
        #     print(i,'click !!')
        #     driver.execute_script('window.scrollTo(0, 4000)')
            
        a,b,c,d,e=[],[],[],[],[]

        soup = BeautifulSoup(driver.page_source,"html.parser")

        for _ in soup.find_all('div',class_='d7ed-d4keTB d7ed-OoK3wU'):
            name = _.find('span',class_='d7ed-Vp2Ugh _8511-Zwkt7j undefined d7ed-KXpuoS d7ed-mzOLVa')
            if name != None :
                name = name.text
                if self.name.lower() in name.lower():
                    price= _.find('span',class_='_8511-GpBMYp d7ed-CLUDGW d7ed-AHa8cD d7ed-giDKVr').text
                    sold = _.find('span',class_='undefined d7ed-bm83Kw d7ed-mzOLVa')
                    if sold != None:
                        sold =sold.text
                    img = _.find('img').get('src')
                    link = _.find('a').get('href')
                    a.append(name)
                    b.append(price)
                    c.append(sold)
                    d.append(link)
                    e.append(img)
        self.df = pd.DataFrame(list(zip(a,b,c,d,e)),columns=['Name','Price','Sold','Link','Img'])
        driver.close()

    
    
