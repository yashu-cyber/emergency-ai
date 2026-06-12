# Hybrid Incident Response Agent

An intelligent, AI-based emergency evacuation system that dynamically routes an agent inside a multi-floor building to the safest and closest exit. Built using **Streamlit**, **NetworkX**, and **Matplotlib**, this dashboard is fully responsive, works offline, and is deployable to Streamlit Cloud.

---

## 🚀 Local Setup & Run

Follow these instructions to run the application on your local machine:

1. **Clone the Repository**
   ```bash
   git clone <repo-url>
   cd hybrid-incident-response-agent
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deploy to Web (Free via Streamlit Cloud)

You can publish this app live on the web in less than 2 minutes using Streamlit Cloud:

1. Push your repository to a public **GitHub** repository.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in using your GitHub account.
3. Click on **"New app"**.
4. Select your repository, set the branch to `main`, and specify the main file path as `app.py`.
5. Click **"Deploy"**. Your application will be live with a shareable URL!

---

## 📂 Project Structure

* **`app.py`**: Entry point of the Streamlit application. Renders the interactive UI, sidebar controls, network map, live metrics, comparison charts, and bottom trade-off panel.
* **`agent/graph.py`**: Defines the building nodes (rooms R1-R10, corridors C1-C5, exits EXIT-A/B/C/D), default edge connections, and manages composite edge weight score calculations (`cost = distance + hazard * 3 + crowd * 2`).
* **`agent/algorithms.py`**: Custom implementation of search algorithms (**A***, **BFS**, and **DFS**) from scratch.
* **`agent/rules.py`**: Evaluates sensor inputs against safety logic thresholds to automatically trigger alarm categories (Red/Orange/Yellow/Green), adjust hazard costs, and block nodes.
* **`agent/reasoner.py`**: Compiles decision logs and trace data into clear step-by-step rationales comparing optimal outcomes.
* **`agent/simulator.py`**: Runs the simulation loop, moves the agent along the selected routing track, and aggregates history metrics.

---

## 🎓 Search Algorithms & Complexity

We implemented all three routing algorithms from scratch without utilizing third-party library solvers:

| Algorithm | Time Complexity | Space Complexity | Optimality | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **A* Search** | $O(E \log V)$ | $O(V)$ | **Optimal** (dynamic composite cost) | Safest path finding during fires, gas leaks, or surges. |
| **BFS** | $O(V + E)$ | $O(V)$ | **Optimal** (hop count only) | Minimizing intermediate corridors traversed regardless of hazards. |
| **DFS** | $O(V + E)$ | $O(V)$ | **Non-optimal** | Path feasibility check, though volatile for risk evasion. |
