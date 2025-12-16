This is a React webapp using the Next.js framework. This webapp was developed on node version `18.17.0`.

## Installation

If you haven't already installed node, please do so from their website. It can be useful to use a node package manager such as `nvm`.

If using `nvm`, use the right node version:

```bash
nvm install 18.17.0
nvm use 18.17.0
```

Next, install dependencies:

```bash
npm install
```

## Getting Started

This is a front-end server which assumes that the backend (flask) server is already running on port `5001`. Make sure that this backend server is already running, and specify its port in `next.config.js`.

Then, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result. The page auto-updates as you edit the file.


## Learn More About Next.js

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
