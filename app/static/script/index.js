const apiUrl = '/names'

async function getNewNames (e) {
    e.preventDefault()
    const form = document.getElementById('form')
    const resContainer = document.getElementById('result')
    if (!form || !resContainer) return

    const data = new FormData(document.getElementById('form'))

    const parameters = {}

    if (data.get('startsWith')) {
        parameters['startsWith'] = data.get('startsWith')
    }

    const fetchUrl = `${apiUrl}?${new URLSearchParams(parameters).toString()}`

    const response = await fetch(fetchUrl, {
        method: 'POST',
        cache: 'no-cache',
        headers: {
            'Content-Type': 'application/json',
        },
    })

    if (response.ok) {
        const json = await response.json()
        resContainer.innerHTML = ''
        for (const name of json.data) {
            const li = document.createElement('li')
            li.innerHTML = name
            resContainer.appendChild(li)
        }

    } else {
        alert('Something went wrong :(')
    }
}

function pageLoadHandler () {
    const btn = document.getElementById('form')
    if (btn) {
        btn.addEventListener('submit', getNewNames)
    }
}

window.addEventListener('load', pageLoadHandler)