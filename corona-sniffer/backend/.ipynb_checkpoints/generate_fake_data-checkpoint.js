/* eslint-disable import/no-extraneous-dependencies */
const cryptoRandomString = require('crypto-random-string');
const randomNormal = require('random-normal');
const db = require('./db');

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

const CENTER = {
  // Helsinki, Finland
  latitude: 60.1628855,
  longitude: 24.94375
};
const SCALE_METERS = 1500;
const RESOLVE_PROB = 0.05;
const N_WALKS = 300;
const WALK_LENGTH = 100;
const TIME_PERIOD = 10;
const THRESHOLD = 30 * TIME_PERIOD;

function linspace(min, max, num) {
  const a = [];
  for (let i = 0; i < num; ++i) {
    a.push(min + (max - min) / (num - 1) * i);
  }
  return a;
}

function randomWalk(startXY, walker) {
  let { x, y } = startXY;

  scaleMeters = SCALE_METERS * 0.2;
  const SIGMA_V = scaleMeters / 100;
  const V_DAMP = 0.03;

  let vx = randomNormal({ dev: 10 * SIGMA_V });
  let vy = randomNormal({ dev: 10 * SIGMA_V });

  const walk = [];

  for (let i = 0; i < WALK_LENGTH; ++i) {
    vx = randomNormal({ mean: vx, dev: SIGMA_V }) * (1 - V_DAMP);
    vy = randomNormal({ mean: vy, dev: SIGMA_V }) * (1 - V_DAMP);
    x += vx * TIME_PERIOD;
    y += vy * TIME_PERIOD;
    walk.push({
      walker: walker,
      time: i,
      x: x,
      y: y
    });
  }
  return walk;
}

const coords = new CoordinateTransforms(CENTER);
function randomId() {
  return cryptoRandomString({ length: 10 });
}

function generateWalkers() {
  const walkers = [];

  for (let i = 0; i < N_WALKS; ++i) {
    walkers.push({
      id: randomId(),
      resolved: Math.random() < RESOLVE_PROB
    });
  }
  return walkers;
}

function generateAgents() {
  const agents = [];
  const s = SCALE_METERS * 0.5;
  const N_PER_ROW = 20;
  const RANGE_M = 30;

  /* Generate a 20x20 grid where 400 agents(devices) are deployed. */
  linspace(-s, s, N_PER_ROW).forEach((x) => {
    linspace(-s, s, N_PER_ROW).forEach((y) => {
      agents.push({
        id: randomId(),
        location: coords.enu2wgs(x, y),
        range: RANGE_M
      });
    });
  });
  return agents;
}

function generateReports(agents, walkers) {
  const N_WALKS = walkers.length;

  const contacts = [];
  const resolvedMap = {};
  let totalHits = 0;
  let totalBroadcasts = 0;

  const walks = [];

  /* For 300 people */
  for (let i = 0; i < N_WALKS; ++i) {
    /* Only 5% of them are infected, thus their IDs are known. */
    const walker = walkers[i];
    const resolved = walker.resolved;
    const resolvedId = walker.id;
    if (resolved) {
      resolvedMap[resolvedId] = [];
    }

    /* Generate the walking path of the person. */
    const walk = randomWalk({
      x: randomNormal({ dev: SCALE_METERS }),
      y: randomNormal({ dev: SCALE_METERS })
    }, walker);

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

      agents
        .filter((agent) => coords.distance(loc, agent.location) < agent.range)
        .forEach((agent) => {
          anyHits = true;
          contacts.push({
            id,
            agent
          });
        });
      if (anyHits) nHits++;
    });
    totalHits += nHits;
    totalBroadcasts += walk.length;
  }

  const coverRate = Math.round(totalHits / totalBroadcasts * 100);
  console.log(`simulated ${totalBroadcasts} broadcasts and generated `
    + `${contacts.length} contact(s). Cover rate ${coverRate}%`);

  return { contacts, resolved: resolvedMap, walks };
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
  const walkers = generateWalkers();
  const { contacts, resolved, walks } = generateReports(generateAgents(), walkers);
  const contact_pairs = generateContactPairs(walks);
  await db.clearAll();

  await Promise.all(contacts.map((contact) => db.insert({
    rollingId: contact.id,
    contactJson: {},
    agentId: contact.agent.id,
    agentJson: contact.agent
  })));
  console.log('inserted contacts');

  await Promise.all(walks.map((walk) => db.insertWalk({
    walkerId: walk.walker.id,
    resolved: walk.walker.resolved,
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
