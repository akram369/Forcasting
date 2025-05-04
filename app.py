The search didn’t return any SHAP or st.pyplot() usage in your current app.py, so let me help you proactively update your SHAP summary plot code to avoid the deprecation warning.

Replace any code like this:

python
Copy
Edit
shap.summary_plot(shap_values, X)
st.pyplot()
With this thread-safe and future-proof version:

python
Copy
Edit
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)
plt.clf()  # clear the figure to avoid memory issues
