#!/bin/bash
set -e

DB="${MONGO_INITDB_DATABASE:-assistml}"

# Username and password from ENV
USER="${MONGO_ASSISTML_USERNAME:-assistml_user}"
PASS="${MONGO_ASSISTML_PASSWORD:-securepassword}"

echo "📦 Initialize MongoDB-Database '$DB' for user '$USER'..."

# Create user with Mongo Shell
mongo "$DB" -u "$MONGO_INITDB_ROOT_USERNAME" -p "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase "admin" <<EOF
db.createUser({
  user: "$USER",
  pwd: "$PASS",
  roles: [ { role: "readWrite", db: "$DB" } ]
});
EOF

echo "✅ User '$USER' created."