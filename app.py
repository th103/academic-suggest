import datetime
from flask import Flask, jsonify, render_template, request, redirect, url_for,session,g
from pymongo import MongoClient
from bson import ObjectId
import torch
from transformers import AutoTokenizer, AutoModel
import fasttext
import numpy as np
from scipy.spatial.distance import cosine
import spacy



ftext_model = fasttext.load_model("nlp_models/cc.en.300.bin")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
sbert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
nlp = spacy.load("en_core_web_sm")
app = Flask(__name__)
app.secret_key="abcd"

client = MongoClient('localhost', 27017)
db = client['academicsuggest']
articles_data = list(db.articles.find())   
 
@app.route('/', methods=['POST','GET'])
def login():
    if g.user:
        return redirect(url_for('home'))
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        session.pop("user",None)
        email = request.form['email']
        password = request.form['password']
        user = db['users'].find_one({'email': email, 'password': password})
        if user:
            session["user"]=request.form["email"]
            return redirect(url_for("home"))
    
    return render_template('index.html')

@app.route('/register', methods=['POST','GET'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        password = request.form['password']
        birth_year = request.form['birth_year']
        gender = request.form['gender']
        location = request.form['location']
        interests = request.form['interests'].split(',')
        
        user = db['users'].find_one({'email': email})
        if user:
            return "Bu e-posta zaten kayıtlı!"
        
        for i, interest in enumerate(interests):
            interests[i] = interest.strip()

        user_data = {
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'password': password,
            'birth_year': birth_year,
            'gender': gender,
            'location': location,
            'interests': interests
        }
        db['users'].insert_one(user_data)
        return render_template('index.html')
    elif request.method == 'GET':
        return render_template('register.html')

@app.route("/profile", methods=['GET', 'POST'])
def profile():
    if g.user: 
        if request.method == 'GET':
            user = db['users'].find_one({'email': session["user"]})
            return render_template('profile.html',
                                   user=session['user'],
                                   first_name=user['first_name'], 
                                   last_name=user['last_name'], 
                                   email=user['email'], 
                                   password=user['password'], 
                                   birth_year=user['birth_year'], 
                                   gender=user['gender'], 
                                   location=user['location'], 
                                   interests=', '.join(user['interests']))
        
        elif request.method == 'POST':
            new_first_name = request.form['first_name']
            new_last_name = request.form['last_name']
            new_email = request.form['email']
            new_password = request.form['password']
            new_birth_year = request.form['birth_year']
            new_gender = request.form['gender']
            new_location = request.form['location']
            new_interests = request.form['interests'].split(',')
            
            user = db['users'].find_one({'email': new_email})
            if user and user["email"]!=session["user"]:
                return redirect(url_for('profile'))

            db['users'].update_one({'email': session['user']}, {
                '$set': {
                    'first_name': new_first_name,
                    'last_name': new_last_name,
                    'email': new_email,
                    'password': new_password,
                    'birth_year': new_birth_year,
                    'gender': new_gender,
                    'location': new_location,
                    'interests': [interest.strip() for interest in new_interests]
                }
            })

            return render_template('profile.html',
                                   user=new_email,
                                   first_name=new_first_name, 
                                   last_name=new_last_name, 
                                   email=new_email, 
                                   password=new_password, 
                                   birth_year=new_birth_year, 
                                   gender=new_gender, 
                                   location=new_location, 
                                   interests=', '.join([interest.strip() for interest in new_interests]))


    return redirect(url_for('login'))

@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route("/home")
def home():
    if 'user' in session:
        articles0 = suggestFasttext(session["user"])
        articles1 = suggestScibert(session["user"])
        #print("-----------------------------------------------------")

        
        return render_template("home.html", user=session["user"], articles0=articles0,articles1=articles1)   
    return redirect(url_for("index"))


@app.route("/stats", methods=["GET"])
def stats():
    if 'user' in session:
        fasttext_likes = 0.0
        fasttext_dislikes = 0.0
        scibert_likes = 0.0
        scibert_dislikes = 0.0
        
        fasttext_likes = db['reactions'].count_documents({"email": session["user"], "model": "0", "react": "like"})
        fasttext_dislikes = db['reactions'].count_documents({"email": session["user"], "model": "0", "react": "dislike"})
        scibert_likes = db['reactions'].count_documents({"email": session["user"], "model": "1", "react": "like"})
        scibert_dislikes = db['reactions'].count_documents({"email": session["user"], "model": "1", "react": "dislike"})
        
        if fasttext_likes + fasttext_dislikes > 0:
            fasttext_precision = round(fasttext_likes / (fasttext_likes + fasttext_dislikes), 2)
        else:
            fasttext_precision = 0.0
            
        if scibert_likes + scibert_dislikes > 0:
            scibert_precision = round(scibert_likes / (scibert_likes + scibert_dislikes), 2)
        else:
            scibert_precision = 0.0

        return render_template("stats.html", user=session["user"], fasttext_likes=fasttext_likes, fasttext_dislikes=fasttext_dislikes, 
                               scibert_likes=scibert_likes, scibert_dislikes=scibert_dislikes, 
                               scibert_precision=scibert_precision, fasttext_precision=fasttext_precision)   
    return redirect(url_for("index"))


@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if g.user: 
        if request.method == 'GET':
            return render_template('upload.html',user=session['user'])
        elif request.method == 'POST':
            title = request.form['title']
            abstract = request.form['abstract']
            
            abstract = abstract.replace("\n", "")
            doc = nlp(abstract)
            clean_abstract =  ' '.join([token.lemma_.lower() for token in doc if token.text != '-' and not token.is_punct and not token.is_stop])
            
            scibert_sentence_vector=sbert_model(torch.tensor(tokenizer.encode(clean_abstract)).unsqueeze(0))[0].mean(1)[0].tolist()
            fasttext_sentence_vector = ftext_model.get_sentence_vector(clean_abstract).tolist()
            
            db['articles'].insert_one({
                'title': title, 
                'abstract': abstract,
                'fasttext_vector': fasttext_sentence_vector,
                'scibert_vector':scibert_sentence_vector,
                "uploader":session['user']})

            return render_template('upload.html',user=session['user'])

    return redirect(url_for('login'))



@app.route('/article/<string:str_id>')
def article(str_id):
    if 'user' in session:
        try:
            article_id = ObjectId(str_id)
        except:
            return "404"
        db['history'].update_one({"email": session["user"], "article_id": article_id},{"$set": {"timestamp": datetime.datetime.now(tz=datetime.timezone.utc)}},upsert=True)
        article = db["articles"].find_one({"_id": article_id})
        if article:
            return render_template('article.html', user=session["user"], article=article)
        else:
            return "404"
    return redirect(url_for("index"))

@app.route("/reaction/<_id>/<model>/<react>", methods=['POST'])
def reaction(_id,model,react):
    if request.method=="POST" and 'user' in session:
        _id=ObjectId(_id)
        react=react
        model=model
        isArticleExist = db['articles'].find_one({'_id': _id})
        if isArticleExist:
            isReactExist = db['reactions'].find_one({"email": session["user"], "article_id": _id, "model": model})
            if isReactExist:
                db['reactions'].delete_one({"email": session["user"], "article_id": _id, "model": model })
                if str(isReactExist["react"]) != react:
                    db['reactions'].insert_one({"email": session["user"], "article_id": _id, "model": model,"react": react,"timestamp": datetime.datetime.now(tz=datetime.timezone.utc)})
            else:
                db['reactions'].insert_one({"email": session["user"], "article_id": _id, "model": model,"react": react,"timestamp": datetime.datetime.now(tz=datetime.timezone.utc)})          
        return jsonify({"message": "Reaction is OK"}), 200
    else:
        return redirect(url_for("index"))

@app.route('/search', methods=['GET'])
def search():
    if request.method == "GET" and 'user' in session:
        title_text = request.args.get('title')
        keyword_text = request.args.get('keyword')
        results = []
        if title_text and keyword_text:
            cursor = db.articles.find({
                "$and": [
                    { "title": { "$regex": title_text, "$options": "i" } },
                    { "abstract": { "$regex": keyword_text, "$options": "i" } }
                ]
            }).limit(20)
            results = list(cursor)
        elif title_text:
            cursor = db.articles.find({
                "$and": [
                    { "title": { "$regex": title_text, "$options": "i" } }
                ]
            }).limit(20)
            results = list(cursor)
        elif keyword_text:
            cursor = db.articles.find({
                "$and": [{ "abstract": { "$regex": keyword_text, "$options": "i" } }]
            }).limit(20)
            results = list(cursor)
        
        """print(str(results))"""
        return render_template('search.html', articles=results,title_text=title_text,keyword_text=keyword_text,user=session["user"])
        
@app.route("/logout")
def logout():
    session.pop("user",None)
    return render_template('index.html')

@app.before_request
def before_request():
    g.user=None

    if "user" in session:
        g.user=session["user"]


def suggestFasttext(email):
    likeVector = likeFasttextVectorCalculator(email)
    historyVector = historyFasttextVectorCalculator(email)
    interestVector = interestsFasttextVectorCalculator(email)
    """
    print("------------------------------------------------------------")
    print("\nlikevec: "+str(likeVector))
    print("\nhisvec: "+str(historyVector))
    print("\nintvec: "+str(interestVector))"""
    
    sum=0.00
    netVector = np.zeros(300,float)
    if likeVector.any():
        sum+=0.333333
        netVector+=likeVector*0.333333
    if historyVector.any():
        sum+=0.333333
        netVector+=historyVector*0.333333
    if interestVector.any():
        sum+=0.333333
        netVector+=interestVector*0.333333
    
    if sum>0:
        netVector=netVector/sum
    
    #print("\nnetvec ftext: "+str(netVector))
   
    similarity_data = list(articles_data)
    reactions_results = list(db['reactions'].find({"email": session["user"]}))
    
    for article in similarity_data:
        for reaction in reactions_results:
            if reaction["article_id"] == article["_id"]:
                article["react"] = reaction["react"]
                
    historyIds = [item['article_id'] for item in list(db["history"].find({"email": session["user"]}))]

    for item in similarity_data:
        if (item.get("react") not in ["like", "dislike"] )and (item["_id"] not in historyIds):
            similarity = float(1 - cosine(np.array(item["fasttext_vector"],float), netVector))
            #print(similarity)
            item["similarity"] = similarity
        else:
            item["similarity"] = 0
            
    similarity_data.sort(key=lambda x: x["similarity"], reverse=True)


    return similarity_data[:5]
    
def likeFasttextVectorCalculator(email):

    likes = list(db["reactions"].find({"email": email,"react":"like","model":"0"}).sort("timestamp", -1).limit(10))
    
    articles_data_f = list(articles_data)
    for like in likes:
        for article in articles_data_f:
            if like["article_id"]==article["_id"]:
                like["fasttext_vector"]=article["fasttext_vector"]
                       
    if len(likes)>=3:
        sum_vector = 0
        k=0.9
        kSum=0
        count=1
        for like in likes:
            sentence_vector = np.array(list(like["fasttext_vector"]),dtype=float)
            new_k=k**count
            kSum+=new_k
            sum_vector = sum_vector + (new_k) * sentence_vector
            count+=1
        likeVector=sum_vector/kSum
        return likeVector
    else:
        return np.array([],dtype=float)

def historyFasttextVectorCalculator(email):
    history = list(db["history"].find({"email": email}).sort("timestamp", -1).limit(10))
    
    articles_data_f=list(articles_data)
    
    for his in history:
        for article in articles_data_f:
            if his["article_id"]==article["_id"]:
                his["fasttext_vector"]=article["fasttext_vector"]
    
    if len(history)>=3:
        sum_vector = 0
        k=0.9
        kSum=0
        count=1
        for h in history:
            sentence_vector = np.array(list(h["fasttext_vector"]),dtype=float)
            new_k=k**count
            kSum+=new_k
            sum_vector=sum_vector+(new_k)*sentence_vector
            count+=1
        historyVector=sum_vector/kSum
        return historyVector
    else:
        return np.array([],dtype=float)
    
def interestsFasttextVectorCalculator(email):
    user_data = db["users"].find_one({"email": email})
    interests = user_data["interests"]
    if len(user_data)>0 and len(interests)>0:
        sum_vector = 0
        for interest in interests:
            sentence_vector = ftext_model.get_sentence_vector(interest)
            sum_vector += sentence_vector     
        average_vector = sum_vector / len(interests)
        return average_vector
    else:
        return np.array([],dtype=float) 




def suggestScibert(email):
    
    likeVector = likeScibertVectorCalculator(email)
    historyVector = historyScibertVectorCalculator(email)
    interestVector = interestsScibertVectorCalculator(email)

    """
    print("\nlikevec: "+str(likeVector))
    print("\nhisvec: "+str(historyVector))
    print("\nintvec: "+str(interestVector))
    """
    
    sum=0.00
    netVector = np.zeros(768,float)
    if likeVector.any():
        sum+=0.333333
        netVector+=likeVector*0.333333
    if historyVector.any():
        sum+=0.333333
        netVector+=historyVector*0.333333
    if interestVector.any():
        sum+=0.333333
        netVector+=interestVector*0.333333
    
    if sum>0:
        netVector=netVector/sum
    
    #print("\nnetvec sbert: "+str(netVector))
    
    similarity_data = list(articles_data)
    reactions_results = list(db['reactions'].find({"email": session["user"]}))
    
    for article in similarity_data:
        for reaction in reactions_results:
            if reaction["article_id"] == article["_id"]:
                article["react"] = reaction["react"]
                

    historyIds = [item['article_id'] for item in list(db["history"].find({"email": session["user"]}))]


    for item in similarity_data:
        if (item.get("react") not in ["like", "dislike"]) and (item["_id"] not in historyIds):
            similarity = float(1 - cosine(np.array(item["scibert_vector"],float), netVector))
            item["similarity"] = similarity
        else:
            item["similarity"] = 0
            
    
            
    similarity_data.sort(key=lambda x: x["similarity"], reverse=True)

    return similarity_data[:5]
    
def likeScibertVectorCalculator(email):

    likes = list(db["reactions"].find({"email": email,"react":"like","model":"1"}).sort("timestamp", -1).limit(10))
    
    
    articles_data_l=list(articles_data)
    
    for like in likes:
        for article in articles_data_l:
            if like["article_id"]==article["_id"]:
                like["scibert_vector"]=article["scibert_vector"]
                       
    if len(likes)>=3:
        sum_vector = 0
        k=0.9
        kSum=0
        count=1
        for like in likes:
            sentence_vector_0 = np.array(list(like["scibert_vector"]),dtype=float)
            new_k=k**count
            kSum+=new_k
            sum_vector = sum_vector + (new_k * sentence_vector_0)
            count+=1
        likeVector=sum_vector/kSum
        return likeVector
    else:
        return np.array([],dtype=float)

def historyScibertVectorCalculator(email):
    history = list(db["history"].find({"email": email}).sort("timestamp", -1).limit(10))
    
    articles_data_h=list(articles_data)
    
    for his in history:
        for article in articles_data_h:
            if his["article_id"]==article["_id"]:
                his["scibert_vector"]=article["scibert_vector"]
    
    if len(history)>=3:
        sum_vector = 0
        k=0.9
        kSum=0
        count=1
        for h in history:
            sentence_vector = np.array(list(h["scibert_vector"]),dtype=float)
            new_k=k**count
            kSum+=new_k
            sum_vector=sum_vector+(new_k)*sentence_vector
            count+=1
        historyVector=sum_vector/kSum
        return historyVector
    else:
        return np.array([],dtype=float)
    
def interestsScibertVectorCalculator(email):
    user_data = db["users"].find_one({"email": email})
    interests = user_data["interests"]
    if len(user_data)>0 and len(interests)>0:
        sum_vector = 0
        for interest in interests:
            sentence_vector = np.array(sbert_model(torch.tensor(tokenizer.encode(interest)).unsqueeze(0))[0].mean(1)[0].tolist(),dtype=float)
            sum_vector += sentence_vector     
        average_vector = sum_vector / len(interests)
        return average_vector
    else:
        return np.array([],dtype=float) 





if __name__ == '__main__':
    #app.run(debug=True)
    app.run(use_reloader=False)
    
    

