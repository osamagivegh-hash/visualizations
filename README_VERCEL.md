# 🚀 Advanced Insurance Analytics Dashboard - Vercel Deployment

## 📋 Overview
This is an advanced insurance analytics dashboard with AI-powered predictions, built with Dash and Plotly, optimized for Vercel deployment.

## ✨ Features
- **10 Interactive Visualizations** with dark theme
- **AI-Powered Predictions** for insurance costs and risk levels
- **Advanced Filtering** with dropdown controls
- **Authentication System** (admin/admin123)
- **Responsive Design** with beautiful styling
- **Real-time Analytics** and insights

## 🚀 Quick Deploy to Vercel

### Option 1: Deploy with Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Follow the prompts and deploy
```

### Option 2: Deploy via GitHub
1. Fork this repository
2. Connect your GitHub account to Vercel
3. Import this repository in Vercel
4. Deploy automatically

## ⚙️ Environment Variables
Set these in your Vercel dashboard:

```env
DB_HOST=your-database-host
DB_NAME=insurance_db
DB_USER=your-username
DB_PASSWORD=your-password
DB_PORT=3306
AUTH_USERNAME=admin
AUTH_PASSWORD=admin123
```

## 📁 Project Structure
```
├── app_vercel.py          # Main Vercel-optimized app
├── vercel.json           # Vercel configuration
├── requirements-vercel.txt # Python dependencies
├── env.example           # Environment variables template
└── README_VERCEL.md      # This file
```

## 🔧 Local Development
```bash
# Install dependencies
pip install -r requirements-vercel.txt

# Run locally
python app_vercel.py
```

## 📊 Dashboard Features
- **Analytics**: Line charts, bar charts, scatter plots, pie charts
- **ML Predictions**: Input customer details and get instant predictions
- **Data Filtering**: Filter by categories, dates, and values
- **Performance Metrics**: Model accuracy and feature importance

## 🎯 Usage
1. **Login** with admin/admin123
2. **Explore** interactive visualizations
3. **Use ML Predictions** section for AI-powered insights
4. **Filter data** using the control panel

## 🛠️ Customization
- Modify `app_vercel.py` for app logic
- Update `vercel.json` for deployment settings
- Adjust environment variables for your database

## 📈 Performance
- Optimized for Vercel's serverless environment
- Efficient data loading and caching
- Responsive design for all devices

## 🔒 Security
- Environment variables for sensitive data
- Authentication system
- Secure database connections

## 📞 Support
For issues or questions, please open an issue in the repository.

---
**Built with ❤️ using Dash, Plotly, and Vercel**
