const sqlite3 = require('sqlite3').verbose();

function createTablesIfNeeded(db) {
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      db
        .run('PRAGMA journal_mode = WAL')
        .run('PRAGMA synchronous = NORMAL')
        .run(`
        CREATE TABLE IF NOT EXISTS agents (
          id text PRIMARY KEY UNIQUE,
          json text NOT NULL
        )`)
        .run(`
        CREATE TABLE IF NOT EXISTS contacts (
          rolling_id text NOT NULL,
          agent_id text NOT NULL,
          walker_id text NOT NULL,
          time int NOT NULL,
          json text NOT NULL,
          resolved_id text,
          distance float
        )`)
        .run(`
        CREATE TABLE IF NOT EXISTS walkers (
          id text PRIMARY KEY,
          id_time int NOT NULL,
          real_id int NOT NULL,
          resolved bool NOT NULL
        )`)
        .run(`
        CREATE TABLE IF NOT EXISTS walks (
          walker_id text NOT NULL,
          walk_time text NOT NULL,
          x float NOT NULL,
          y float NOT NULL,
          json text NOT NULL,
          PRIMARY KEY (walker_id, walk_time),
          FOREIGN KEY (walker_id) REFERENCES walkers(real_id)
        )`)
        .run(`
          CREATE TABLE IF NOT EXISTS velocitys (
          walker_id text NOT NULL,
          time int NOT NULL,
          x_json text NOT NULL,
          y_json text NOT NULL,
          ratio float NOT NULL,
          PRIMARY KEY (walker_id, time),
          FOREIGN KEY (walker_id) REFERENCES walkers(real_id)
        )`)
        .run(`
        CREATE TABLE IF NOT EXISTS pairs (
          id1 text NOT NULL,
          id2 text NOT NULL,
          time int NOT NULL,
          distance float NOT NULL,
          PRIMARY KEY (id1, id2, time)
        )`)
        .run(`
        CREATE INDEX IF NOT EXISTS index_rolling_id
          ON contacts(rolling_id)`)
        .run(`
        CREATE INDEX IF NOT EXISTS index_resolved_id
          ON contacts(resolved_id)
          WHERE resolved_id IS NOT NULL
        `, (err) => (err ? reject(err) : resolve(true)));
    });
  });
  console.log('created tables');
}

const validated = {
  id(str) {
    const MAX_ID_LENGTH = 100;
    if (!str) throw new Error('missing ID');
    if (str.length > MAX_ID_LENGTH) throw new Error('ID max length exceeded');
    return str;
  },
  json(obj) {
    const MAX_JSON_LENGTH = 2000;
    const str = JSON.stringify(obj);
    if (str.length > MAX_JSON_LENGTH) throw new Error('JSON max length exceeded');
    return str;
  }
};

function databaseApi(db) {
  function promiseRun(...args) {
    // console.log(...args);
    return new Promise((resolve, reject) => db.run(...args,
      (err) => (err ? reject(err) : resolve(true))));
  }

  return {
    insertAgent({
      agentId, agentJson
    }) {
      return Promise.all([
        promiseRun(`
          INSERT INTO agents (id, json) VALUES (?, ?) ON CONFLICT(id) DO NOTHING`,
        [
          validated.id(agentId),
          validated.json(agentJson)
        ])
      ]);
    },
    insert({
      rollingId, contactJson, agentId, walkerId, agentJson, time, distance
    }) {
      return Promise.all([
        promiseRun(`
          INSERT INTO contacts (rolling_id, agent_id, walker_id, time, json, distance) VALUES (?, ?, ?, ?, ?, ?)`,
        [
          validated.id(rollingId),
          validated.id(agentId),
          validated.id(walkerId),
          time,
          validated.json(contactJson),
          distance,
        ])
      ]);
    },
    insertWalker({
      id, time, resolved, real_id
    }) {
      return Promise.all([
        promiseRun(`
          INSERT INTO walkers (id, id_time, resolved, real_id) VALUES (?, ?, ?, ?)`,
        [
          validated.id(id),
          time,
          resolved,
          real_id
        ])
      ]);
    },
    insertWalk({
      walkerId, time, x, y, json
    }) {
      return Promise.all([
        promiseRun(`
          INSERT INTO walks (walker_id, walk_time, x, y, json) VALUES (?, ?, ?, ?, ?) ON CONFLICT(walker_id, walk_time) DO NOTHING`,
        [
          validated.id(walkerId),
          time,
          x,
          y,
          validated.json(json)
        ])
      ]);
    },
    insertV({
      walkerId, time, x_json, y_json, ratio
    }) {
      return Promise.all([
        promiseRun(`
          INSERT INTO velocitys (walker_id, time, x_json, y_json, ratio) VALUES (?, ?, ?, ?, ?) ON CONFLICT(walker_id, time) DO NOTHING`,
        [
          validated.id(walkerId),
          time,
          validated.json(x_json),
          validated.json(y_json),
          ratio
        ])
      ]);
    },
    insertPair({
      id1, id2, time, distance
    }) {
      return Promise.all([
        promiseRun(`
          INSERT INTO pairs (id1, id2, time, distance) VALUES (?, ?, ?, ?)`,
        [
          validated.id(id1),
          validated.id(id2),
          time, 
          distance
        ]),
      ]);
    },
    updateResolved({ resolvedId, rollingIds }) {
      const inList = rollingIds.map(() => '?').join(',');
      return promiseRun(`
        UPDATE contacts SET resolved_id = ? WHERE rolling_id IN (${inList})`,
      [validated.id(resolvedId)].concat(rollingIds.map(validated.id)));
    },
    getResolved(each, finalize) {
      db.each(`
        SELECT * FROM contacts
        INNER JOIN agents ON contacts.agent_id = agents.id
        WHERE contacts.resolved_id IS NOT NULL
      `,
      (err, result) => {
        if (err) throw err;
        each(result);
      }, finalize);
    },
    getAll(cb, finalize) {
      db.each(`
        SELECT * FROM contacts
        INNER JOIN agents ON contacts.agent_id = agents.id
      `,
      (err, result) => {
        if (err) throw err;
        cb(result);
      }, finalize);
    },
    getAgents(each, finalize) {
      db.each(`
        SELECT * FROM agents
      `,
      (err, result) => {
        if (err) throw err;
        each(result);
      }, finalize);
    },
    getWalks(each, finalize) {
      db.each(`
        SELECT * FROM walks
      `,
      (err, result) => {
        if (err) throw err;
        each(result);
      }, finalize);
    },
    getWalksAttached(each, finalize) {
      db.each(`
        SELECT * FROM walks_attached
      `,
      (err, result) => {
        if (err) throw err;
        each(result);
      }, finalize);
    },
    getWalksGT(each, finalize) {
      db.each(`
        SELECT * FROM walks
        INNER JOIN walkers ON walkers.id = walks.walker_id
      `,
      (err, result) => {
        if (err) throw err;
        each(result);
      }, finalize);
    },
    clearAll() {
      console.log('clearing database');
      return Promise.all([
        'DROP TABLE contacts',
        'DROP TABLE agents',
        'DROP TABLE walkers',
        'DROP TABLE walks',
        'DROP TABLE pairs',
        'DROP TABLE IF EXISTS walks_attached',
        'DROP TABLE velocitys'
      ].map((stm) => promiseRun(stm)))
        .then(() => createTablesIfNeeded(db));
    },
    transaction(cb) {
      db.serialize(() => {
        cb(this);
      });
    }
  };
}

function createAndConnect() {
  const dbFile = 'data/database_100+_train_2.db';
  const db = new sqlite3.Database(dbFile);
  createTablesIfNeeded(db);
  return databaseApi(db);
}

module.exports = createAndConnect();
