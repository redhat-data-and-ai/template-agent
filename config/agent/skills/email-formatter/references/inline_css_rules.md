# Gmail-Compatible Inline CSS Rules

## Core Constraint

Gmail strips `<style>` blocks and CSS classes. **All styling must be inline.**

## Required Inline Styles by Element

### Container
```html
<div style="max-width:600px;margin:0 auto;font-family:Arial,Helvetica,sans-serif;color:#333;">
```

### Header Banner
```html
<div style="background-color:#CC0000;color:white;padding:20px;text-align:center;">
  <h1 style="margin:0;font-size:22px;">...</h1>
</div>
```

### Metric Highlight Box
```html
<div style="background-color:#f5f5f5;padding:12px;border-radius:6px;margin-bottom:16px;">
```

### Section Heading
```html
<h3 style="color:#CC0000;margin-top:24px;">...</h3>
```

### Section Divider
```html
<hr style="border:none;border-top:1px solid #ddd;margin:20px 0;">
```

### Footer Disclaimer
```html
<div style="padding:12px 20px;font-size:12px;color:#999;text-align:center;">
```

## Layout

- Use `<table>` with inline styles for multi-column layout
- Maximum width: **600px** (mobile email client compatibility)
- Content padding: 20px on body sections

## Color Palette

- Red Hat primary: `#CC0000`
- Text primary: `#333`
- Text muted: `#999`
- Background highlight: `#f5f5f5`
- Border/divider: `#ddd`
