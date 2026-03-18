#!/bin/bash
# setup.sh — run this once to set up the full project

echo "Setting up AI Services Dashboard..."

# create folder structure
mkdir -p full_stack_project/dashboard

# copy dashboard HTML
cp index.html full_stack_project/dashboard/index.html
cp docker-compose.yml full_stack_project/docker-compose.yml

echo ""
echo "Folder structure needed:"
echo ""
echo "full_stack_project/"
echo "├── docker-compose.yml"
echo "├── dashboard/"
echo "│   └── index.html"
echo "├── llm-router/     (unzip llm-router.zip here)"
echo "├── rag-app/        (unzip rag-app.zip here)"
echo "├── chat-app/       (unzip chat-app.zip here)"
echo "├── youtube-app/    (unzip youtube-app.zip here)"
echo "└── research-app/   (unzip research-app.zip here)"
echo ""
echo "Then edit each app's .env with your API keys."
echo ""
echo "Then run: docker compose up --build"
echo ""
echo "Dashboard opens at: http://localhost:3000"
