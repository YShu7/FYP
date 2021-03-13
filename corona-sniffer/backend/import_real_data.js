const fs = require('fs');
const db = require('./db');

// const PATH = '../../datasets/cabspottingdata/tmp';
const PATH = '../../datasets/Geolife/tmp';

async function generateDb() {
	await db.clearAll();

	await fs.readFile(PATH + '/agents.txt', 'utf8', (err, jsonString) => {
	    if (err) {
	        console.log("Error reading file from disk:", err)
	        return
	    }
	    try {
	        const agents = JSON.parse(jsonString)
	        Promise.all(agents.map((agent) => db.insertAgent({
	        	agentId: agent.id,
				agentJson: agent
			})));
			console.log('inserted agents');
		} catch(err) {
	        console.log('Error parsing JSON string:', err)
	    }
	})

	await fs.readFile(PATH + '/contacts.txt', 'utf8', (err, jsonString) => {
	    if (err) {
	        console.log("Error reading file from disk:", err)
	        return
	    }
	    try {
	        const contacts = JSON.parse(jsonString)
	        Promise.all(contacts.map((contact) => db.insert({
				rollingId: contact.id,
				contactJson: contact.json,
				distance: contact.distance,
				agentId: contact.agent.id,
				walkerId: contact.walker.id,
				agentJson: contact.agent,
				time: contact.time
			})));
			console.log('inserted contacts');
		} catch(err) {
	        console.log('Error parsing JSON string:', err)
	    }
	})

	await fs.readFile(PATH + '/walkers.txt', 'utf8', (err, jsonString) => {
	    if (err) {
	        console.log("Error reading file from disk:", err)
	        return
	    }
	    try {
	        const walkers = JSON.parse(jsonString);
	        Promise.all(walkers.map((walker) => db.insertWalker({
				id: walker.id,
				time: walker.time,
				resolved: walker.resolved,
				real_id: walker.real_id
			})));
			console.log('inserted walkers');
		} catch(err) {
	        console.log('Error parsing JSON string:', err)
	    }
	})

	await fs.readFile(PATH + '/walks.txt', 'utf8', (err, jsonString) => {
	    if (err) {
	        console.log("Error reading file from disk:", err)
	        return
	    }
	    try {
	        const walks = JSON.parse(jsonString)
	        Promise.all(walks.map((walk) => db.insertWalk({
				walkerId: walk.walker.id,
				time: walk.time,
				x: walk.x,
				y: walk.y,
				json: {
				  resolved: walk.walker.resolved,
				  location: {
				  	latitude: walk.latitude, 
				  	longitude: walk.longitude
				  }
				}
			})))
			console.log('inserted walks');
		} catch(err) {
	        console.log('Error parsing JSON string:', err)
	    }
	})

	await fs.readFile(PATH + '/contact_pairs.txt', 'utf8', (err, jsonString) => {
	    if (err) {
	        console.log("Error reading file from disk:", err)
	        return
	    }
	    try {
	        const contact_pairs = JSON.parse(jsonString)
			Promise.all(contact_pairs.map((pair) => db.insertPair({
				id1: pair.id1,
				id2: pair.id2,
				time: pair.time,
				distance: pair.distance
			})))
			console.log('inserted pairs');
		} catch(err) {
	        console.log('Error parsing JSON string:', err)
	    }
	})

	await fs.readFile(PATH + '/resolved.txt', 'utf8', (err, jsonString) => {
	    if (err) {
	        console.log("Error reading file from disk:", err)
	        return
	    }
	    try {
	        const resolved = JSON.parse(jsonString)
	        Promise.all(
			Object.entries(resolved)
				.map(([resolvedId, rollingIds]) => db.updateResolved({ resolvedId, rollingIds })));
			console.log('done');
		} catch(err) {
	        console.log('Error parsing JSON string:', err)
	    }
	})
}

generateDb();
