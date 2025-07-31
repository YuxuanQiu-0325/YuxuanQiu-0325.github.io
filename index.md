---
layout: default
---

<div style="width: 80%; margin: 0 auto;">
  <h1>我的博客</h1>
  <p>欢迎来到我的技术博客，这里记录了我的学习和思考。</p>

  <hr>

  <h2>文章列表</h2>
  <ul>
    {% for post in site.posts %}
      <li>
        <strong><a href="{{ post.url }}">{{ post.title }}</a></strong>
        <span style="color: #666; margin-left: 10px;">{{ post.date | date: "%Y-%m-%d" }}</span>
      </li>
    {% endfor %}
  </ul>
</div>
