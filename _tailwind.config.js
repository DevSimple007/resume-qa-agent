/** Tailwind 编译配置 —— 把 static/index.html 用到的样式编译成 static/vendor/tailwind.css */
module.exports = {
  content: ['./static/index.html', './static/admin.html'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'PingFang SC',
               'Hiragino Sans GB', 'Microsoft YaHei', 'Helvetica Neue', 'Arial', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      boxShadow: {
        'paper': '0 1px 3px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.03)',
        'float': '0 12px 32px -4px rgba(0, 0, 0, 0.06)',
      },
    },
  },
}
