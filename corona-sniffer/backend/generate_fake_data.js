/* eslint-disable import/no-extraneous-dependencies */
const cryptoRandomString = require('crypto-random-string');
const randomNormal = require('random-normal');
const db = require('./db');
const fs = require('fs');

const ONLY_IN_AREA = true;
const CENTER = {
  // Helsinki, Finland
  latitude: 60.1628855,
  longitude: 24.94375
};
const SCALE_METERS = 1500;
const RESOLVE_PROB = 0.05;
const N_WALKS = 10;
const INTERVAL = 60;
const WALK_LENGTH = 150;
const TIME_PERIOD = 1;
const N_PER_ROW = 30;
const RANGE_M = SCALE_METERS / (N_PER_ROW - 1);
const THRESHOLD = RANGE_M;
const UPDATE_TIME = 10 * INTERVAL;
const CURR_TIME = Math.floor(Date.now() / 1000);

function CoordinateTransforms(center) {
  const EARTH_R = 6.371e6;
  const METERS_PER_LAT = Math.PI * EARTH_R / 180.0;
  const metersPerLon = METERS_PER_LAT * Math.cos(center.latitude / 180.0 * Math.PI);

  this.wgs2enu = (latitude, longitude) => ({
    x: (longitude - center.longitude) * metersPerLon,
    y: (latitude - center.latitude) * METERS_PER_LAT
  });

  this.enu2wgs = (x, y) => ({
    latitude: y / METERS_PER_LAT + center.latitude,
    longitude: x / metersPerLon + center.longitude
  });

  const wgs2enuObj = ({ latitude, longitude }) => this.wgs2enu(latitude, longitude);

  this.distance = (wgs0, wgs1) => {
    const enu0 = wgs2enuObj(wgs0);
    const enu1 = wgs2enuObj(wgs1);
    return Math.sqrt((enu1.x - enu0.x) ** 2 + (enu1.y - enu0.y) ** 2);
  };
}

function linspace(min, max, num) {
  const a = [];
  for (let i = 0; i < num; ++i) {
    a.push(min + (max - min) / (num - 1) * i);
  }
  return a;
}

function randomWalk(startXY, walker_id) {
  let { x, y } = startXY;

  scaleMeters = SCALE_METERS * 0.2;
  const SIGMA_V = scaleMeters / 100;
  const V_DAMP = 0.03;

  let vx = randomNormal({ dev: 10 * SIGMA_V });
  let vy = randomNormal({ dev: 10 * SIGMA_V });

  const walk = [];
  const walker = [];
  let curr_time = CURR_TIME - Math.floor(Math.random() * 15 * 60);
  const resolved = Math.random() < RESOLVE_PROB;

  for (let i = 0; i < (WALK_LENGTH / (UPDATE_TIME / INTERVAL)) + 2; i++) {
    this_walker = {
      real_id: walker_id,
      id: randomId(),
      resolved: resolved,
      time: curr_time + i * UPDATE_TIME
    };
    walker.push(this_walker);
  }

  let this_walker_id = 0;
  for (let i = 0; i < WALK_LENGTH / TIME_PERIOD; ++i) {
    let time = CURR_TIME + i * INTERVAL * TIME_PERIOD;
    let this_walker = walker[this_walker_id];
    // console.log(i, this_walker, this_walker_id, this_walker.time + UPDATE_TIME, time);
    while (this_walker.time + UPDATE_TIME < time) {
      this_walker = walker[++this_walker_id];
    }
  	
    const meanx = vx;
    const meany = vy;
    vx = randomNormal({ mean: meanx, dev: SIGMA_V }) * (1 - V_DAMP);
    vy = randomNormal({ mean: meany, dev: SIGMA_V }) * (1 - V_DAMP);
    x += vx * INTERVAL * TIME_PERIOD / 60;
    y += vy * INTERVAL * TIME_PERIOD / 60;

    walk.push({
      walker: this_walker,
      time: time,
      x: x,
      y: y
    });
  }
  return { walk, walker, resolved };
}

const coords = new CoordinateTransforms(CENTER);
function randomId() {
  return cryptoRandomString({ length: 20 });
}

function generateAgents() {
  const agents = [];
  const s = SCALE_METERS * 0.5;

  /* Generate a 20x20 grid where 400 agents(devices) are deployed. */
  let idx = 0;
  let idy = 0;
  linspace(-s, s, N_PER_ROW).forEach((x) => {
    linspace(-s, s, N_PER_ROW).forEach((y) => {
      agents.push({
        id: idx + 'EVEN' + idy,
        location: coords.enu2wgs(x, y),
        position: {x: x, y: y},
        range: RANGE_M
      });
      idy += 1;
    });
    idx += 1;
  });

  // idx = 0;
  // idy = 0;
  // linspace(-s+RANGE_M, s-RANGE_M, N_PER_ROW-1).forEach((x) => {
  //   linspace(-s+RANGE_M, s-RANGE_M, N_PER_ROW-1).forEach((y) => {
  //     agents.push({
  //       id: idx + 'ODD' + idy,
  //       location: coords.enu2wgs(x, y),
  //       position: {x: x, y: y},
  //       range: RANGE_M
  //     });
  //     idy += 1;
  //   });
  //   idx += 1;
  // });

  return agents;
}

function generateReports(agents) {
  const contacts = [];
  const resolvedMap = {};
  let totalHits = 0;
  let totalBroadcasts = 0;

  const walks = [];
  const walkers = [];

  /* For 300 people */
  let i = 0;
  while (i < N_WALKS) {
    /* Only 5% of them are infected, thus their IDs are known. */

    /* Generate the walking path of the person. */
    const { walk, walker, resolved } = randomWalk({
      x: randomNormal({ dev: SCALE_METERS }),
      y: randomNormal({ dev: SCALE_METERS })
    }, i);

    let numInArea = 0;
    if (ONLY_IN_AREA) {
      walk.forEach(( walk ) => {
        if (walk.x < -SCALE_METERS/2 || walk.x > SCALE_METERS/2 ||
          walk.y < -SCALE_METERS/2 || walk.y > SCALE_METERS/2) {
          
        } else {
          numInArea += 1;
        }
      });
      if (numInArea < WALK_LENGTH) {
        continue;
      }
    }

    const resolvedId = i;
    if (resolved) {
    	resolvedMap[resolvedId] = [];
		}

		walker.forEach(( walker ) => {
			walkers.push(walker);
		});

    /* When the person is in the detection range of any device, he is tracked. */
    let nHits = 0;
    walk.forEach(( walk ) => {
      let x = walk.x;
      let y = walk.y;
      const loc = coords.enu2wgs(x, y);
      const id = randomId();

      walks.push(walk);

      /* If he is a infected person, keep this contact pair's id. */
      if (resolved) resolvedMap[resolvedId].push(id);
      let anyHits = false;

      filtered_agents = agents
        .filter((agent) => coords.distance(loc, agent.location) < agent.range)
      //   .sort((a, b) => {
      //     return coords.distance(loc, a.location) - coords.distance(loc, b.location)
      //   })
      // if (filtered_agents.length >= 1) {
      //   anyHits = true;
      //   let agent = filtered_agents[0];
      //   contacts.push({
      //     id: id,
      //     agent: agent, 
      //     walker: walk.walker,
      //     time: walk.time,
      //     json: {
      //       distance: coords.distance(loc, agent.location),
      //       agentPos: coords.wgs2enu(agent.location.latitude, agent.location.longitude)
      //     }
      //   })
      // }
        .forEach((agent) => {
          anyHits = true;
          contacts.push({
            id: id,
            agent: agent, 
            walker: walk.walker,
            time: walk.time,
            json: {
              distance: coords.distance(loc, agent.location),
              agentPos: coords.wgs2enu(agent.location.latitude, agent.location.longitude)
            }
          });
        });
      if (anyHits) nHits++;
    });

    totalHits += nHits;
    totalBroadcasts += walk.length;
    i++;
  }

  const coverRate = Math.round(totalHits / totalBroadcasts * 100);
  console.log(`simulated ${totalBroadcasts} broadcasts and generated `
    + `${contacts.length} contact(s). Cover rate ${coverRate}%`);
  return { contacts, resolved: resolvedMap, walks, walkers };
}

function generateContactPairs(walks) {
  const pairs = [];
  let nPairs = 0;
  walks.forEach(( walk ) => {
    let x1 = walk.x;
    let y1 = walk.y;
    const loc1 = coords.enu2wgs(x1, y1);

    walks
      .filter(( walk2 ) => walk2.walker.id != walk.walker.id && walk2.time == walk.time)
      .forEach(( walk2 ) => {
        let x2 = walk2.x;
        let y2 = walk2.y;
        const loc2 = coords.enu2wgs(x2, y2);
        const distance = coords.distance(loc1, loc2);
        if (distance < THRESHOLD) {
          pairs.push({
            id1: walk.walker.id,
            id2: walk2.walker.id,
            time: walk.time,
            distance: distance
          });
          nPairs++;
        }
      });
  });
  const coverRate = Math.round(nPairs / (N_WALKS * N_WALKS) * 100);
  console.log(`simulated ${walks.length} walks, ${N_WALKS} walkers and generated `
    + `${nPairs} pair(s). Cover rate ${coverRate}%`);

  return pairs;
}

async function generateDb() {
  const agents = generateAgents();
  const { contacts, resolved, walks, walkers } = generateReports(agents);
  const contact_pairs = generateContactPairs(walks);
  await db.clearAll();

  await Promise.all(agents.map((agent) => db.insertAgent({
    agentId: agent.id,
    agentJson: agent
  })));
  await Promise.all(contacts.map((contact) => db.insert({
    rollingId: contact.id,
    contactJson: contact.json,
    distance: contact.json.distance,
    agentId: contact.agent.id,
    walkerId: contact.walker.id,
    agentJson: contact.agent,
    time: contact.time
  })));
  console.log('inserted contacts');

  await Promise.all(walkers.map((walker) => db.insertWalker({
  	id: walker.id,
    time: walker.time,
  	resolved: walker.resolved,
  	real_id: walker.real_id
  })));
  console.log('inserted walkers');

  await Promise.all(walks.map((walk) => db.insertWalk({
    walkerId: walk.walker.id,
    time: walk.time,
    x: walk.x,
    y: walk.y,
    json: {
      resolved: walk.walker.resolved,
      location: coords.enu2wgs(walk.x, walk.y)
    }
  })))
  console.log('inserted walks');

  await Promise.all(contact_pairs.map((pair) => db.insertPair({
    id1: pair.id1,
    id2: pair.id2,
    time: pair.time,
    distance: pair.distance
  })))
  console.log('inserted pairs');

  await Promise.all(
    Object.entries(resolved)
      .map(([resolvedId, rollingIds]) => db.updateResolved({ resolvedId, rollingIds }))
  );
  console.log('done');
}

generateDb();
