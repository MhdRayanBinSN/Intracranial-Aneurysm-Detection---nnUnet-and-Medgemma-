FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source
COPY . .

# Expose dev port
EXPOSE 5173

# Start dev server with host binding
CMD ["npm", "run", "dev", "--", "--host"]
