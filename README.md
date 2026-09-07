# SpectraEdge

**SpectraEdge: A Multi-Scale Multi-Object Edge Detection and Frequency-Domain Analysis System**

This is the website project for the university Signals and Linear Systems project.

## Open the website

On Windows, double-click **Start Website.cmd** in this folder. Keep the terminal open while using the website. Open the local address printed in the terminal (normally http://localhost:3000).

Or run from this folder:

```powershell
npm.cmd run dev
```

On macOS/Linux, use `npm run dev`. Node.js 22.13 or newer is required. No Python or desktop application installation is needed for this GUI phase.

For a fresh copy, install the frontend dependencies once:

```powershell
npm.cmd run setup
```

## Where things belong

```text
SpectraEdge/
├── frontend/             Website: React, TypeScript, styles, and browser interactions
│   ├── app/              Analyze, Compare, and Live routes
│   ├── components/       Controls, image inspector, pipeline, and shared session
│   ├── lib/              Parameters, demo data, and the future API result contract
│   ├── public/           Website icons and images
│   ├── tests/            Input and parameter validation checks
│   ├── package.json      Frontend dependencies and commands
│   └── README.md         Frontend architecture and integration notes
├── backend/              Reserved for the future Python analysis API
│   └── README.md         Backend scope and connection plan
├── Start Website.cmd     Double-click to run locally on Windows
├── package.json          Convenient commands from this project folder
└── README.md             Start here
```

The entire project uses one Git repository at this root, including the frontend, backend folder, and startup files. The original website history is preserved. Hosting configuration remains under `frontend/.openai/`; local development does not publish or update a hosted website.

## Current phase

Analyze, Compare, and Live are interactive GUI pages. Local image selection, inspection, zoom, pan, exports, controls, and preferences work. All edge, gradient, contour, FFT, and comparison previews are labeled demo data.

The processing backend and webcam capture are not implemented yet. Uploaded images stay in the browser and are not sent to a server. The `backend/` folder is preparation, not a second application you need to start.

## Useful commands

Run these from this project folder. Use `npm.cmd` in Windows PowerShell if its execution policy blocks `npm`.

| Command | Purpose |
| --- | --- |
| `npm run setup` | Install frontend dependencies |
| `npm run dev` | Start the local development website |
| `npm run build` | Create a production build locally |
| `npm start` | Serve an existing production build |
| `npm run check` | Run lint, TypeScript checks, and validation tests |

Stop a running website with **Ctrl+C** in its terminal. If port 3000 is already occupied, use the local URL printed by the server.
