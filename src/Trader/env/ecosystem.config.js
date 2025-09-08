module.exports = {
  apps: [
    {
      name: "stock-server",
      cmd: "src/Trader/manage.py",
      args: "run --no-reload",
      autorestart: false,
      watch: false,
      interpreter: "/root/anaconda3/envs/stock-bot-env/bin/python",
      env: {
        ENV: "prod",
      },
      env_production: {
        ENV: "prod",
      },
    },
  ],
};
