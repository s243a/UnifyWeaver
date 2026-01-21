/**
 * Authentication Module for HTTP CLI Server
 *
 * Provides JWT-based authentication with role-based access control.
 *
 * @module unifyweaver/shell/auth
 */

import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// Configuration
// ============================================================================

const JWT_SECRET = process.env.JWT_SECRET || 'unifyweaver-dev-secret-change-in-production';
const TOKEN_EXPIRY = 24 * 60 * 60; // 24 hours in seconds

// Users file location (relative to server working directory)
const USERS_FILE = process.env.USERS_FILE || path.join(process.cwd(), 'users.json');

// ============================================================================
// Types
// ============================================================================

export interface User {
  id: string;
  email: string;
  passwordHash: string;
  roles: string[];
  permissions: string[];
  createdAt: string;
}

export interface TokenPayload {
  sub: string;
  email: string;
  roles: string[];
  permissions: string[];
  iat: number;
  exp: number;
}

export interface AuthResult {
  success: boolean;
  token?: string;
  user?: Omit<User, 'passwordHash'>;
  error?: string;
}

// ============================================================================
// Password Hashing (SHA256 with salt)
// ============================================================================

export function hashPassword(password: string): string {
  const salt = crypto.randomBytes(16).toString('hex');
  const hash = crypto.createHash('sha256').update(password + salt).digest('hex');
  return `sha256:${salt}:${hash}`;
}

export function verifyPassword(password: string, storedHash: string): boolean {
  if (storedHash.startsWith('sha256:')) {
    const [, salt, hash] = storedHash.split(':');
    const testHash = crypto.createHash('sha256').update(password + salt).digest('hex');
    return hash === testHash;
  }
  // Fallback for plain passwords (dev only)
  return password === storedHash;
}

// ============================================================================
// JWT Implementation
// ============================================================================

function base64UrlEncode(data: object): string {
  return Buffer.from(JSON.stringify(data))
    .toString('base64')
    .replace(/=/g, '')
    .replace(/\+/g, '-')
    .replace(/\//g, '_');
}

function base64UrlDecode(str: string): object {
  str = str.replace(/-/g, '+').replace(/_/g, '/');
  while (str.length % 4) str += '=';
  return JSON.parse(Buffer.from(str, 'base64').toString());
}

export function createToken(user: Omit<User, 'passwordHash' | 'createdAt'>): string {
  const header = { alg: 'HS256', typ: 'JWT' };
  const now = Math.floor(Date.now() / 1000);

  const payload: TokenPayload = {
    sub: user.id,
    email: user.email,
    roles: user.roles,
    permissions: user.permissions,
    iat: now,
    exp: now + TOKEN_EXPIRY
  };

  const headerB64 = base64UrlEncode(header);
  const payloadB64 = base64UrlEncode(payload);
  const signature = crypto
    .createHmac('sha256', JWT_SECRET)
    .update(`${headerB64}.${payloadB64}`)
    .digest('base64')
    .replace(/=/g, '')
    .replace(/\+/g, '-')
    .replace(/\//g, '_');

  return `${headerB64}.${payloadB64}.${signature}`;
}

export function verifyToken(token: string): TokenPayload | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;

    const [headerB64, payloadB64, signature] = parts;

    const expectedSig = crypto
      .createHmac('sha256', JWT_SECRET)
      .update(`${headerB64}.${payloadB64}`)
      .digest('base64')
      .replace(/=/g, '')
      .replace(/\+/g, '-')
      .replace(/\//g, '_');

    if (signature !== expectedSig) {
      return null;
    }

    const payload = base64UrlDecode(payloadB64) as TokenPayload;

    if (payload.exp && payload.exp < Math.floor(Date.now() / 1000)) {
      return null; // Token expired
    }

    return payload;
  } catch {
    return null;
  }
}

// ============================================================================
// User Storage
// ============================================================================

interface UsersDb {
  users: User[];
}

function loadUsers(): UsersDb {
  if (!fs.existsSync(USERS_FILE)) {
    // Create default users
    const defaultUsers: UsersDb = {
      users: [
        {
          id: 'shell',
          email: 'shell@local',
          passwordHash: hashPassword('shell'),
          roles: ['shell', 'admin', 'user'],
          permissions: ['read', 'write', 'delete', 'shell'],
          createdAt: new Date().toISOString()
        },
        {
          id: 'admin',
          email: 'admin@local',
          passwordHash: hashPassword('admin'),
          roles: ['admin', 'user'],
          permissions: ['read', 'write', 'delete'],
          createdAt: new Date().toISOString()
        },
        {
          id: 'user',
          email: 'user@local',
          passwordHash: hashPassword('user'),
          roles: ['user'],
          permissions: ['read'],
          createdAt: new Date().toISOString()
        }
      ]
    };
    fs.writeFileSync(USERS_FILE, JSON.stringify(defaultUsers, null, 2));
    console.log(`Created default users file: ${USERS_FILE}`);
    console.log('Default users: shell@local/shell, admin@local/admin, user@local/user');
    return defaultUsers;
  }

  return JSON.parse(fs.readFileSync(USERS_FILE, 'utf-8'));
}

function saveUsers(db: UsersDb): void {
  fs.writeFileSync(USERS_FILE, JSON.stringify(db, null, 2));
}

export function findUser(email: string): User | null {
  const db = loadUsers();
  return db.users.find(u => u.email.toLowerCase() === email.toLowerCase()) || null;
}

export function findUserById(id: string): User | null {
  const db = loadUsers();
  return db.users.find(u => u.id === id) || null;
}

// ============================================================================
// Auth Operations
// ============================================================================

export function login(email: string, password: string): AuthResult {
  const user = findUser(email);

  if (!user) {
    return { success: false, error: 'User not found' };
  }

  if (!verifyPassword(password, user.passwordHash)) {
    return { success: false, error: 'Invalid password' };
  }

  const token = createToken({
    id: user.id,
    email: user.email,
    roles: user.roles,
    permissions: user.permissions
  });

  const { passwordHash, ...safeUser } = user;
  return { success: true, token, user: safeUser };
}

export function register(email: string, password: string, roles: string[] = ['user']): AuthResult {
  if (findUser(email)) {
    return { success: false, error: 'User already exists' };
  }

  const db = loadUsers();
  const id = email.split('@')[0].toLowerCase();

  const newUser: User = {
    id,
    email: email.toLowerCase(),
    passwordHash: hashPassword(password),
    roles,
    permissions: roles.includes('admin') ? ['read', 'write', 'delete'] : ['read'],
    createdAt: new Date().toISOString()
  };

  db.users.push(newUser);
  saveUsers(db);

  const token = createToken({
    id: newUser.id,
    email: newUser.email,
    roles: newUser.roles,
    permissions: newUser.permissions
  });

  const { passwordHash, ...safeUser } = newUser;
  return { success: true, token, user: safeUser };
}

export function getUserFromToken(token: string): AuthResult {
  const payload = verifyToken(token);

  if (!payload) {
    return { success: false, error: 'Invalid or expired token' };
  }

  return {
    success: true,
    user: {
      id: payload.sub,
      email: payload.email,
      roles: payload.roles,
      permissions: payload.permissions,
      createdAt: ''
    }
  };
}

// ============================================================================
// Role Checking
// ============================================================================

export function hasRole(user: { roles: string[] }, role: string): boolean {
  return user.roles.includes(role);
}

export function hasAnyRole(user: { roles: string[] }, roles: string[]): boolean {
  return roles.some(role => user.roles.includes(role));
}

export function canAccessShell(user: { roles: string[] }): boolean {
  return hasRole(user, 'shell');
}

export function canExecuteCommands(user: { roles: string[] }): boolean {
  return hasAnyRole(user, ['shell', 'admin']);
}

export function canBrowse(user: { roles: string[] }): boolean {
  return hasAnyRole(user, ['shell', 'admin', 'user']);
}
