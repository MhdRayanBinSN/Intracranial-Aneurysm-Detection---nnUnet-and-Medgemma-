FROM node:18-alpine

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Expose Vite ports
EXPOSE 5173
EXPOSE 5174

# Run the Vite development server exposing it to the network
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
