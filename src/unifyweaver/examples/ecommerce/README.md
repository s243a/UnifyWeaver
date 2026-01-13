# E-commerce Example App

A full-featured e-commerce application demonstrating all UnifyWeaver UI patterns working together.

## Overview

This example shows how to build a complete e-commerce app using declarative patterns that compile to multiple frontend frameworks with Python backend support.

## Features

| Feature | Patterns Used |
|---------|---------------|
| Product Catalog | `infinite_list`, `data(query)` |
| Product Detail | `navigation(stack)`, `data(query)` |
| Product Search | `form_field(search)`, `filter_store` |
| Shopping Cart | `state(global, cartStore)` |
| Cart Persistence | `persistence(local, cart)` |
| Checkout Form | `form_pattern`, `validation` |
| User Login | `login_flow`, `data(mutation)` |
| User Registration | `register_flow`, `data(mutation)` |
| Order History | `infinite_list`, `data(query)` |
| User Preferences | `persistence(local, userPrefs)` |

## Architecture

### Frontend Targets

| Target | Framework | Navigation | State | Data Fetching |
|--------|-----------|------------|-------|---------------|
| React Native | TypeScript | React Navigation | Zustand | React Query |
| Vue 3 | TypeScript | Vue Router | Pinia | Vue Query |
| Flutter | Dart | GoRouter | Riverpod | http/dio |
| SwiftUI | Swift | NavigationStack | ObservableObject | URLSession |

### Backend Targets

| Target | Framework | Validation | ORM |
|--------|-----------|------------|-----|
| FastAPI | Python 3.9+ | Pydantic | SQLAlchemy |
| Flask | Python 3.9+ | Custom | Flask-SQLAlchemy |
| Express | Node.js | Zod | Prisma |
| Go | Go 1.21+ | go-playground | GORM |

## Usage

### Interactive

```prolog
% Load the module
?- [ecommerce_app].

% Define patterns
?- define_ecommerce_app.

% Generate React Native project
?- generate_ecommerce_project(react_native, '/output/ecommerce-rn').

% Generate Vue + FastAPI full stack
?- generate_ecommerce_full_stack(vue, fastapi, '/output/ecommerce-vue').

% Generate Flutter + Flask full stack
?- generate_ecommerce_full_stack(flutter, flask, '/output/ecommerce-flutter').

% Generate all frontend targets
?- generate_ecommerce_all_frontends('/output/ecommerce').
```

### Generated Project Structure

**React Native Frontend:**
```
ecommerce-rn/
├── package.json
├── tsconfig.json
├── App.tsx
├── src/
│   ├── components/
│   │   ├── ProductCard.tsx
│   │   ├── CartItem.tsx
│   │   └── ...
│   ├── screens/
│   │   ├── HomeScreen.tsx
│   │   ├── ProductScreen.tsx
│   │   ├── CartScreen.tsx
│   │   └── ...
│   ├── stores/
│   │   ├── cartStore.ts
│   │   ├── userStore.ts
│   │   └── filterStore.ts
│   ├── hooks/
│   │   ├── useProducts.ts
│   │   └── useOrders.ts
│   └── api/
│       └── client.ts
```

**FastAPI Backend:**
```
backend/
├── requirements.txt
├── main.py
├── app/
│   ├── routes/
│   │   ├── products.py
│   │   ├── orders.py
│   │   ├── cart.py
│   │   └── auth.py
│   ├── models/
│   ├── schemas/
│   └── services/
```

## Pattern Definitions

### Navigation

```prolog
% Tab navigation with 4 screens
navigation(tab, [
    screen(home, 'HomeScreen', [icon('home')]),
    screen(search, 'SearchScreen', [icon('search')]),
    screen(cart, 'CartScreen', [icon('cart')]),
    screen(profile, 'ProfileScreen', [icon('user')])
], [])
```

### Shopping Cart State

```prolog
state(global, [
    store(cartStore),
    slices([
        slice(cart, [
            field(items, 'CartItem[]'),
            field(total, number),
            field(itemCount, number)
        ], [
            action(addItem, "..."),
            action(removeItem, "..."),
            action(updateQuantity, "..."),
            action(clearCart, "...")
        ])
    ])
], [])
```

### Data Fetching

```prolog
% Products query with pagination
data(query, [
    name(fetchProducts),
    endpoint('/api/products'),
    stale_time(60000),
    retry(3)
])

% Create order mutation
data(mutation, [
    name(createOrder),
    endpoint('/api/orders'),
    method('POST'),
    invalidates([fetch_orders])
])
```

### Checkout Form

```prolog
form_pattern(checkout_form, [
    field(email, email, [required], []),
    field(firstName, text, [required, min_length(2)], []),
    field(lastName, text, [required, min_length(2)], []),
    field(address, text, [required], []),
    field(city, text, [required], []),
    field(zipCode, text, [required, pattern('^[0-9]{5}$')], []),
    field(cardNumber, text, [required, pattern('^[0-9]{16}$')], []),
    field(cardExpiry, text, [required], []),
    field(cardCvv, password, [required, min_length(3), max_length(4)], [])
], _)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/products` | List products (paginated) |
| GET | `/api/products/:id` | Get product details |
| GET | `/api/search` | Search products |
| GET | `/api/categories` | List categories |
| GET | `/api/cart` | Get user's cart |
| POST | `/api/cart/items` | Add item to cart |
| PUT | `/api/cart/items/:id` | Update cart item |
| DELETE | `/api/cart/items/:id` | Remove from cart |
| POST | `/api/orders` | Create order |
| GET | `/api/orders` | List user orders |
| GET | `/api/orders/:id` | Get order details |
| POST | `/api/auth/login` | User login |
| POST | `/api/auth/register` | User registration |
| POST | `/api/auth/forgot-password` | Password reset |

## Testing

```bash
# Run e-commerce app tests (43 tests)
swipl -g "run_tests" -t halt test_ecommerce_app.pl

# Run inline tests
swipl -g "test_ecommerce_app" -t halt ecommerce_app.pl
```

## Files

| File | Description |
|------|-------------|
| `ecommerce_app.pl` | App specification and generation |
| `test_ecommerce_app.pl` | 43 plunit tests |
| `README.md` | This documentation |

## Dependencies

This example uses:
- `ui_patterns.pl` - Core UI patterns
- `ui_patterns_extended.pl` - Extended patterns (forms, lists, modals, auth)
- `pattern_glue.pl` - Frontend/backend integration
- `project_generator.pl` - File output system
- `fastapi_generator.pl` - FastAPI backend generation
- `flask_generator.pl` - Flask backend generation
