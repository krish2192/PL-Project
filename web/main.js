(function () {
    const apiBaseInput = document.getElementById("apiBase");
    const homeSel = document.getElementById("home");
    const awaySel = document.getElementById("away");
    const dateInput = document.getElementById("matchDate");
    const resultCard = document.getElementById("result");
    const numbersDiv = document.getElementById("numbers");
    const topClassP = document.getElementById("topClass");
    const statusLine = document.getElementById("status");
    const loadBtn = document.getElementById("loadTeamsBtn");
    const predictBtn = document.getElementById("predictBtn");


    // default to local API
    apiBaseInput.value = apiBaseInput.value || "http://127.0.0.1:8000";

    function setStatus(msg, isError = false) {
        statusLine.textContent = msg || "";
        statusLine.className = isError ? "error" : "muted";
    }

    async function loadTeams() {
        const base = apiBaseInput.value.trim().replace(/\/$/, "");
        if (!base) { setStatus("Enter API base URL"); return; }
        setStatus("Loading teams…");
        try {
            const res = await fetch(`${base}/teams`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const teams = await res.json();
            if (!Array.isArray(teams) || teams.length < 2) throw new Error("Bad /teams response");
            homeSel.innerHTML = ""; awaySel.innerHTML = "";
            teams.forEach(t => {
                const h = document.createElement("option");
                h.value = t; h.textContent = t; homeSel.appendChild(h);
                const a = document.createElement("option");
                a.value = t; a.textContent = t; awaySel.appendChild(a);
            });
            // preselect different teams if possible
            awaySel.selectedIndex = Math.min(1, teams.length - 1);
            setStatus("Teams loaded.");
        } catch (err) {
            console.error(err);
            setStatus("Failed to load teams. Is the API running? Check /health.", true);
        }
    }

    async function loadLimits() {
        const base = apiBaseInput.value.trim().replace(/\/$/, "");
        try {
            const res = await fetch(`${base}/limits`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const lim = await res.json();
            dateInput.min = lim.min;
            dateInput.max = lim.max;
            // set default if empty or out of range
            if (!dateInput.value || dateInput.value < lim.min || dateInput.value > lim.max) {
                dateInput.value = lim.default;
            }
        } catch (e) {
            console.error(e);
            // leave date as-is; predict() will still error nicely if out of range
        }
    }

    async function predict() {
        const base = apiBaseInput.value.trim().replace(/\/$/, "");
        const body = {
            home_team: homeSel.value,
            away_team: awaySel.value,
            date: dateInput.value || new Date().toISOString().slice(0, 10),
        };
        if (body.home_team === body.away_team) {
            setStatus("Home and away must be different teams.", true);
            return;
        }
        setStatus("Predicting…");
        try {
            const res = await fetch(`${base}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            const ph = (data.home_win * 100).toFixed(1);
            const pd = (data.draw * 100).toFixed(1);
            const pa = (data.away_win * 100).toFixed(1);

            numbersDiv.innerHTML = `Home: <b>${ph}%</b> &nbsp;&nbsp; Draw: <b>${pd}%</b> &nbsp;&nbsp; Away: <b>${pa}%</b>`;
            const total = (data.home_win + data.draw + data.away_win) || 1;
            document.querySelector(".bars").style.setProperty("--homeW", (data.home_win / total) * 100 + "%");
            document.querySelector(".bars").style.setProperty("--drawW", (data.draw / total) * 100 + "%");
            document.querySelector(".bars").style.setProperty("--awayW", (data.away_win / total) * 100 + "%");

            topClassP.textContent = `Most likely: ${data.top_class}`;
            resultCard.style.display = "block";
            setStatus("Done.");
        } catch (err) {
            console.error(err);
            setStatus("Prediction failed: " + err.message, true);
        }
    }



    // wire up
    loadBtn.addEventListener("click", loadTeams);
    predictBtn.addEventListener("click", predict);

    // first load
    loadLimits();
    loadTeams();
})();
