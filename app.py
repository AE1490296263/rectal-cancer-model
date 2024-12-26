from flask import Flask, request, render_template
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# 加载模型
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # 获取用户输入的特征值
        features = [
            float(request.form['CIKP']),
            float(request.form['CEA']),
            float(request.form['ALB']),
            float(request.form['PT']),
            float(request.form['Cyfra211'])
        ]
        
        features_array = np.array([features])
        
        # 预测类别概率
        prediction = model.predict_proba(features_array)[0]
        
        # 生成 SHAP 图
        explainer = shap.Explainer(model)
        shap_values = explainer(features_array)
        
        # 二分类时，可视化第 1 类(下标1)的 SHAP 值
        shap.plots.waterfall(shap_values[0][:, 1], show=False)
        
        plt.savefig('static/shap_plot.png')
        plt.close()
        
        return render_template('result.html', prediction=prediction)
    
    # 默认 GET 请求，渲染 index.html
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
