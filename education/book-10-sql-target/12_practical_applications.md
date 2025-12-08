<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 12: Practical Applications

This chapter demonstrates real-world applications using the SQL target, combining concepts from previous chapters into complete, production-ready solutions.

## Application 1: Sales Dashboard

Build a comprehensive sales analytics dashboard with multiple views.

### Schema

```prolog
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(customers, [id-integer, name-text, email-text, region-text, signup_date-text]).
:- sql_table(products, [id-integer, name-text, category-text, price-real]).
:- sql_table(orders, [id-integer, customer_id-integer, product_id-integer, quantity-integer, amount-real, order_date-text, status-text]).
:- sql_table(salespeople, [id-integer, name-text, region-text, hire_date-text]).
```

### Dashboard Queries

```prolog
%% ============================================
%% Revenue Summary (Monthly)
%% ============================================

monthly_revenue(Year, Month, Revenue, OrderCount, AvgOrder) :-
    orders(_, _, _, _, Amount, OrderDate, Status),
    Status = 'completed',
    Year = sql_extract(year, OrderDate),
    Month = sql_extract(month, OrderDate),
    sql_group_by([Year, Month]),
    Revenue = sql_as(sql_sum(Amount), total_revenue),
    OrderCount = sql_as(sql_count(*), order_count),
    AvgOrder = sql_as(sql_avg(Amount), avg_order_value),
    sql_order_by(Year, desc),
    sql_order_by(Month, desc).

%% ============================================
%% Top Products by Revenue
%% ============================================

top_products(ProductName, Category, Revenue, UnitsSold, Rank) :-
    orders(_, _, ProductId, Qty, Amount, _, Status),
    products(ProductId, ProductName, Category, _),
    Status = 'completed',
    sql_group_by([ProductId, ProductName, Category]),
    Revenue = sql_sum(Amount),
    UnitsSold = sql_sum(Qty),
    Rank = sql_window(rank, [], [], [(Revenue, desc)]),
    sql_order_by(Revenue, desc),
    sql_limit(10).

%% ============================================
%% Customer Segments
%% ============================================

customer_segments(CustomerId, CustomerName, Segment, TotalSpent, OrderCount) :-
    customers(CustomerId, CustomerName, _, _, _),
    orders(_, CustomerId, _, _, Amount, _, Status),
    Status = 'completed',
    sql_group_by([CustomerId, CustomerName]),
    TotalSpent = sql_sum(Amount),
    OrderCount = sql_count(*),
    Segment = sql_case([
        when((TotalSpent > 10000, OrderCount > 20), 'Platinum'),
        when((TotalSpent > 5000, OrderCount > 10), 'Gold'),
        when((TotalSpent > 1000, OrderCount > 5), 'Silver')
    ], 'Bronze').

%% ============================================
%% Regional Performance
%% ============================================

regional_performance(Region, Revenue, Customers, AvgPerCustomer) :-
    customers(CustId, _, _, Region, _),
    orders(_, CustId, _, _, Amount, _, Status),
    Status = 'completed',
    sql_group_by([Region]),
    Revenue = sql_sum(Amount),
    Customers = sql_count_distinct(CustId),
    AvgPerCustomer = Revenue / Customers,
    sql_order_by(Revenue, desc).

%% ============================================
%% Month-over-Month Growth
%% ============================================

:- sql_recursive_table(monthly_sales, [month-text, revenue-real]).

mom_growth(Month, Revenue, PrevRevenue, Growth) :-
    orders(_, _, _, _, Amount, OrderDate, Status),
    Status = 'completed',
    Month = sql_strftime('%Y-%m', OrderDate),
    sql_group_by([Month]),
    Revenue = sql_sum(Amount),
    PrevRevenue = sql_window(lag, [Revenue, 1], [], [(Month, asc)]),
    Growth = sql_case([
        when(sql_is_null(PrevRevenue), 0)
    ], ((Revenue - PrevRevenue) / PrevRevenue) * 100).
```

## Application 2: Employee Management System

### Schema

```prolog
:- sql_table(employees, [id-integer, name-text, dept_id-integer, manager_id-integer, salary-integer, hire_date-text, status-text]).
:- sql_table(departments, [id-integer, name-text, budget-real, head_id-integer]).
:- sql_table(performance_reviews, [id-integer, employee_id-integer, review_date-text, score-integer, reviewer_id-integer]).

:- sql_recursive_table(org_chart, [id-integer, name-text, manager_id-integer, level-integer]).
```

### HR Queries

```prolog
%% ============================================
%% Department Summary
%% ============================================

dept_summary(DeptName, EmpCount, AvgSalary, TotalSalary, BudgetUsed) :-
    departments(DeptId, DeptName, Budget, _),
    employees(_, _, DeptId, _, Salary, _, Status),
    Status = 'active',
    sql_group_by([DeptId, DeptName, Budget]),
    EmpCount = sql_count(*),
    AvgSalary = sql_avg(Salary),
    TotalSalary = sql_sum(Salary),
    BudgetUsed = sql_as((TotalSalary / Budget) * 100, budget_percent).

%% ============================================
%% Organization Hierarchy
%% ============================================

org_base(Id, Name, ManagerId, Level) :-
    employees(Id, Name, _, ManagerId, _, _, Status),
    Status = 'active',
    sql_is_null(ManagerId),
    Level = 0.

org_recursive(Id, Name, ManagerId, Level) :-
    employees(Id, Name, _, ManagerId, _, _, Status),
    Status = 'active',
    org_chart(ManagerId, _, _, ParentLevel),
    Level = ParentLevel + 1.

org_result(Id, Name, ManagerId, Level) :-
    org_chart(Id, Name, ManagerId, Level),
    sql_order_by(Level, asc),
    sql_order_by(Name, asc).

%% ============================================
%% Performance Ranking
%% ============================================

performance_ranking(EmpName, DeptName, AvgScore, DeptRank, CompanyRank) :-
    employees(EmpId, EmpName, DeptId, _, _, _, Status),
    departments(DeptId, DeptName, _, _),
    performance_reviews(_, EmpId, _, Score, _),
    Status = 'active',
    sql_group_by([EmpId, EmpName, DeptId, DeptName]),
    AvgScore = sql_avg(Score),
    DeptRank = sql_window(rank, [], [DeptId], [(AvgScore, desc)]),
    CompanyRank = sql_window(rank, [], [], [(AvgScore, desc)]).

%% ============================================
%% Tenure Analysis
%% ============================================

tenure_analysis(EmpName, HireDate, YearsEmployed, TenureBand) :-
    employees(_, EmpName, _, _, _, HireDate, Status),
    Status = 'active',
    YearsEmployed = sql_date_diff(sql_date('now'), HireDate) / 365,
    TenureBand = sql_case([
        when(YearsEmployed > 10, 'Veteran (10+ yrs)'),
        when(YearsEmployed > 5, 'Experienced (5-10 yrs)'),
        when(YearsEmployed > 2, 'Established (2-5 yrs)'),
        when(YearsEmployed > 1, 'Developing (1-2 yrs)')
    ], 'New (< 1 yr)').

%% ============================================
%% Salary Analysis
%% ============================================

salary_analysis(DeptName, SalaryBand, Count, AvgSalary) :-
    employees(_, _, DeptId, _, Salary, _, Status),
    departments(DeptId, DeptName, _, _),
    Status = 'active',
    SalaryBand = sql_case([
        when(Salary > 150000, '$150K+'),
        when(Salary > 100000, '$100K-$150K'),
        when(Salary > 75000, '$75K-$100K'),
        when(Salary > 50000, '$50K-$75K')
    ], 'Under $50K'),
    sql_group_by([DeptId, DeptName, SalaryBand]),
    Count = sql_count(*),
    AvgSalary = sql_avg(Salary).
```

## Application 3: E-Commerce Analytics

### Schema

```prolog
:- sql_table(users, [id-integer, email-text, created_at-text, last_login-text]).
:- sql_table(sessions, [id-integer, user_id-integer, start_time-text, end_time-text, pages_viewed-integer]).
:- sql_table(cart_items, [id-integer, session_id-integer, product_id-integer, quantity-integer, added_at-text]).
:- sql_table(purchases, [id-integer, user_id-integer, session_id-integer, amount-real, purchased_at-text]).
:- sql_table(products, [id-integer, name-text, category-text, price-real]).
```

### Analytics Queries

```prolog
%% ============================================
%% Conversion Funnel
%% ============================================

conversion_funnel(Date, TotalSessions, WithCart, Converted, ConversionRate) :-
    sessions(SessionId, _, StartTime, _, _),
    Date = sql_date(StartTime),
    sql_group_by([Date]),
    TotalSessions = sql_count_distinct(SessionId),

    % Sessions with cart items
    WithCart = sql_sum(sql_case([
        when(sql_exists(cart_items(_, SessionId, _, _, _)), 1)
    ], 0)),

    % Sessions with purchase
    Converted = sql_sum(sql_case([
        when(sql_exists(purchases(_, _, SessionId, _, _)), 1)
    ], 0)),

    ConversionRate = (Converted * 100.0) / TotalSessions.

%% ============================================
%% Cart Abandonment Analysis
%% ============================================

cart_abandonment(Category, CartsCreated, Abandoned, AbandonRate, LostRevenue) :-
    cart_items(_, SessionId, ProductId, Qty, _),
    products(ProductId, _, Category, Price),
    sql_group_by([Category]),
    CartsCreated = sql_count_distinct(SessionId),

    % Sessions without purchase
    Abandoned = sql_sum(sql_case([
        when(sql_not_exists(purchases(_, _, SessionId, _, _)), 1)
    ], 0)),

    AbandonRate = (Abandoned * 100.0) / CartsCreated,
    LostRevenue = sql_sum(sql_case([
        when(sql_not_exists(purchases(_, _, SessionId, _, _)), Qty * Price)
    ], 0)).

%% ============================================
%% User Cohort Analysis
%% ============================================

cohort_retention(SignupMonth, PurchaseMonth, Users, RetentionRate) :-
    users(UserId, _, CreatedAt, _),
    purchases(_, UserId, _, _, PurchasedAt),
    SignupMonth = sql_strftime('%Y-%m', CreatedAt),
    PurchaseMonth = sql_strftime('%Y-%m', PurchasedAt),
    sql_group_by([SignupMonth, PurchaseMonth]),
    Users = sql_count_distinct(UserId),
    % Would need CTE for full retention calculation
    RetentionRate = 0.  % Placeholder

%% ============================================
%% Product Recommendations (Frequently Bought Together)
%% ============================================

bought_together(Product1, Product2, Frequency) :-
    purchases(PurchaseId1, UserId, _, _, _),
    purchases(PurchaseId2, UserId, _, _, _),
    cart_items(_, _, Prod1Id, _, _),
    cart_items(_, _, Prod2Id, _, _),
    products(Prod1Id, Product1, _, _),
    products(Prod2Id, Product2, _, _),
    Prod1Id < Prod2Id,  % Avoid duplicates
    sql_group_by([Prod1Id, Prod2Id, Product1, Product2]),
    Frequency = sql_count(*),
    sql_having(Frequency >= 5),
    sql_order_by(Frequency, desc),
    sql_limit(20).

%% ============================================
%% RFM Segmentation (Recency, Frequency, Monetary)
%% ============================================

rfm_segmentation(UserId, Email, Recency, Frequency, Monetary, Segment) :-
    users(UserId, Email, _, _),
    purchases(_, UserId, _, Amount, PurchasedAt),
    sql_group_by([UserId, Email]),

    % Days since last purchase
    Recency = sql_date_diff(sql_date('now'), sql_max(PurchasedAt)),
    Frequency = sql_count(*),
    Monetary = sql_sum(Amount),

    Segment = sql_case([
        when((Recency < 30, Frequency > 10, Monetary > 1000), 'Champions'),
        when((Recency < 30, Frequency > 5), 'Loyal'),
        when((Recency < 90, Monetary > 500), 'Potential Loyalist'),
        when((Recency > 180, Frequency > 5), 'At Risk'),
        when(Recency > 365, 'Lost')
    ], 'Regular').
```

## Application 4: Generating Database Views

Create a set of views for a reporting database:

```prolog
%% ============================================
%% Generate Multiple Views
%% ============================================

generate_views :-
    % Revenue summary view
    compile_predicate_to_sql(monthly_revenue/5,
        [format(view), view_name(v_monthly_revenue)], SQL1),
    write_sql_file('views/v_monthly_revenue.sql', SQL1),

    % Customer segments view
    compile_predicate_to_sql(customer_segments/5,
        [format(view), view_name(v_customer_segments)], SQL2),
    write_sql_file('views/v_customer_segments.sql', SQL2),

    % Department summary view
    compile_predicate_to_sql(dept_summary/5,
        [format(view), view_name(v_dept_summary)], SQL3),
    write_sql_file('views/v_dept_summary.sql', SQL3),

    % Generate org chart using recursive CTE
    compile_recursive_cte(org_chart, [id, name, manager_id, level],
        recursive_cte(org_base/4, org_recursive/4),
        org_result/4,
        [view_name(v_org_chart)], SQL4),
    write_sql_file('views/v_org_chart.sql', SQL4),

    format('Generated all views successfully!~n').
```

## Application 5: Data Quality Reports

```prolog
%% ============================================
%% Data Quality Checks
%% ============================================

% Find duplicate emails
duplicate_emails(Email, Count) :-
    users(_, Email, _, _),
    sql_group_by([Email]),
    Count = sql_count(*),
    sql_having(Count > 1).

% Find orphaned records
orphaned_orders(OrderId) :-
    orders(OrderId, CustomerId, _, _, _, _, _),
    sql_not_exists(customers(CustomerId, _, _, _, _)).

% Find null values in critical fields
null_audit(TableName, FieldName, NullCount) :-
    % This would need dynamic SQL generation
    % Simplified version for employees table
    employees(_, Name, DeptId, _, _, _, _),
    sql_group_by([]),
    NullCount = sql_sum(sql_case([when(sql_is_null(Name), 1)], 0)).

% Data freshness check
data_freshness(TableName, LastUpdate, DaysOld, Status) :-
    orders(_, _, _, _, _, OrderDate, _),
    LastUpdate = sql_max(OrderDate),
    DaysOld = sql_date_diff(sql_date('now'), LastUpdate),
    Status = sql_case([
        when(DaysOld < 1, 'Fresh'),
        when(DaysOld < 7, 'Recent'),
        when(DaysOld < 30, 'Stale')
    ], 'Very Stale').
```

## Best Practices Summary

### 1. Query Organization

- Group related queries into logical modules
- Use consistent naming conventions
- Document complex business logic

### 2. Performance

- Use appropriate indexes based on generated JOIN conditions
- Consider materialized views for expensive computations
- Test with production-like data volumes

### 3. Maintainability

- Keep predicates focused and single-purpose
- Use CTEs for complex multi-step logic
- Add comments explaining business rules

### 4. Testing

```prolog
test_all_queries :-
    format('Testing SQL generation...~n'),

    % Test each predicate compiles without error
    forall(
        member(Pred, [
            monthly_revenue/5,
            customer_segments/5,
            dept_summary/5,
            performance_ranking/5
        ]),
        (
            format('Testing ~w... ', [Pred]),
            (compile_predicate_to_sql(Pred, [], _)
             -> format('OK~n')
             ;  format('FAILED~n'))
        )
    ).
```

## Exercises

1. **Sales Dashboard**: Extend the sales dashboard with:
   - Year-over-year comparison
   - Product category breakdown
   - Seasonal trend analysis

2. **Employee System**: Add queries for:
   - Succession planning (employees near retirement)
   - Compensation equity analysis
   - Training needs assessment

3. **E-Commerce**: Build queries for:
   - Customer lifetime value prediction
   - Inventory turnover analysis
   - Price elasticity metrics

4. **Data Quality**: Create a comprehensive data quality report that checks all tables for:
   - Null values in required fields
   - Orphaned foreign keys
   - Data freshness
   - Duplicate detection

## Summary

In this chapter, you've seen how to apply all SQL target features to build real-world applications:

- Sales dashboards with aggregations, window functions, and CASE WHEN
- HR systems with recursive CTEs for org charts
- E-commerce analytics with complex joins and subqueries
- Automated view generation for reporting databases
- Data quality monitoring queries

## Conclusion

Congratulations on completing the SQL Target book! You now have the skills to:

- Generate any SQL query type from Prolog predicates
- Build complete reporting and analytics solutions
- Create maintainable, well-organized query libraries
- Apply best practices for production deployments

Continue exploring by combining SQL target with other UnifyWeaver targets for full-stack data applications.

---

## Navigation

**←** [Previous: Chapter 11: CASE WHEN Expressions](11_case_when) | [📖 Book 10: SQL Target](./) | [Next: Book 11: Prolog Target →](../book-11-prolog-target/)
