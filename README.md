# hy-ai-chatbot-chainlit

## How to Use

1. Install Python 3.13
2. Install packages
   ```
   pip install -r requirements.txt
   ```
3. Run sample codes
   1. Simple Agent
      ```
      cd 1_simple_agent
      chainlit run main.py
      ```
   2. MCP
      ```
      cd 2_mcp
      chainlit run main.py
      ```
   3. Predict Questions
      ```
      cd 3_predict_questions
      chainlit run main.py
      ```
   4. DB
      ```
      cd 4_db
      docker compose -f docker-compose.yaml up -d --build
      ```
