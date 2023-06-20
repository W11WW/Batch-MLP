//
// Created by clash on 10/06/2023.
//
#include <vector>

// for physical commodities
float notionalValueP(const float& currentlyTradedPrice, const float& numberOfUnits)
{
    return numberOfUnits * currentlyTradedPrice;
}

// for a stock index - point value of a stock index or similar contract is set by the exchange. too high - is risky
// too low - transaction costs may be prohibitive
float notionalValueS(const float& currentlyTradedPrice, const float& pointValue)
{
    return currentlyTradedPrice * pointValue;
}

// forward price F with the underlying asset being Physical Commodities (Grains, energy Products, Precious Metals, ect)
// you can rearrange the following equation to obtain the cash price given the contracts price
float forwardPricePC(const float commodityPrice, const float timeToMaturity, const float interestRate, const float annualS, const float annualI)
{
    // annualS = annual storage costs per commodity unit
    // annualI = annual insurance costs per commodity unit
    return commodityPrice * (1 + interestRate * timeToMaturity) + (annualS * timeToMaturity) + (annualI * timeToMaturity);
}

// forward price F with the underlying asset being Stock
float forwardPriceS(const float stockPrice, const float timeToMaturity, const float interestRate, const std::vector<float>& dividendPayments, const std::vector<float>& dividendPaymentsTime,
                    const std::vector<float>& dividendInterestRates)
{
    // dividendPayments = each dividend payment expected prior to maturity of the forward contract
    // dividendPaymentsTime = time remaining to maturity after each dividend payment
    // dividendInterestRates = the applicable interest rate (the forward rate) from each dividend payment to maturity of the forward contract.
    float dividendInterestRateValueSum  = 0.0f;
    for(int i = 0; i < dividendPayments.size(); i++)
    {
        dividendInterestRateValueSum += -(dividendPayments[i] * (1 + dividendInterestRates[i] * dividendPaymentsTime[i]));
    }
    return (stockPrice * (1 + interestRate * timeToMaturity)) + dividendInterestRateValueSum;
}

// the above equation can be rewritten as interest on dividends will we small and so we aggregate all the dividends D expected over the life of the forward contract and
// ignore any interest that can be earned on the dividends.
float forwardPriceS(const float stockPrice, const float timeToMaturity, const float interestRate, const float dividends)
{
    return (stockPrice * (1 + interestRate * timeToMaturity)) - dividends;
}

// forward price F with the underlying asset being Bonds and Notes - similar to stock evaluation if we treat the coupon payments as if they were dividends
float forwardPriceB(const float bondPrice, const float timeToMaturity, const float interestRate, const std::vector<float>& couponPayments, const std::vector<float>& couponPaymentsTime,
                    const std::vector<float>& couponInterestRates)
{
    // couponPayments = each coupon expected prior to maturity of the forward contract
    // couponPaymentsTime = time remaining to maturity after each coupon payment
    // couponInterestRates = the applicable interest rate (the forward rate) from each coupon payment to maturity of the forward contract.
    float couponInterestRateValueSum  = 0.0f;
    for(int i = 0; i < couponPayments.size(); i++)
    {
        couponInterestRateValueSum += -(couponPayments[i] * (1 + couponInterestRates[i] * couponPaymentsTime[i]));
    }
    return (bondPrice * (1 + interestRate * timeToMaturity)) + couponInterestRateValueSum;
}

// forward price F for foreign currency -

// Spot / cash transaction - both parties agree on terms, followed immediately by an exchange of money for goods.
// forward contract - the parties agree on the terms now, but the actual exchange of money for goods does not take place until some later date, the maturity or expiration date.
// futures contract - forward contract traded on an organized exchange.
// option contract - gives one party the right to make a decision at a later date
// put option - gives one party the right to decide whether to sell at a later date.
// call option - gives one party the right to decide whether to buy at a later date.
// premium - This amount is negotiated between the buyer and the seller, and the seller keeps the premium regardless of any subsequent decision on the part of the buyer.
// exercise price - how much the holder will receive if certain events occur.
// derivative contracts / derivatives - forwards, futures, and options.
// swap - an agreement to exchange cash flows.
// opening trade - the first trade to take place, either buying or selling, resulting in an open position.
// closing trade - a subsequent trade, reversing the initial trade.
// open interest - measure of trading activity in exchange-traded derivative contracts, the number of contracts traded on an exchange that have not yet been closed out.
// the number of contracts traded on an exchange that have not yet been closed out must be equal because for every buyer there must be a seller.
// long -  if a trader first buys a contract (opening trade)
// short - if a trader first sells a contract (opening trade)
// long position - when the total trade results in a debit
// short position - when the total trade results in a credit
// notional value / nominal value - different in various scenarios such as physical commodities or financial instruments.
// unrealized / paper profit - profit gained by the value of the stock increasing.
// realized - must go back into the market and sell.
// margin deposit - the exchange wants to ensure that both parties live up to their obligations. to do this the exchange collects
// an amount from each party that it holds as security against possible default by the buyer or seller.
// futures-type settlement - where there is an initial margin deposit followed by daily cash transfers, is also known as margin and variation settlement.
// cash-settled - types of futures, where no physical delivery takes place at maturity.
// variation - is a credit or debit that results from fluctuations in the price of a futures contract.
// forward price = current cash price + costs of buying now - benefits of buying now
// ** traders in forward or futures contracts sometimes refer to the basis, the difference between the cash price and the forward price
// ** a normal or contango commodity market is one in which long-term futures contracts trade at a premium to short-term contracts. But
// sometimes the opposite occurs - a futures contract will trade at a discount to cash. If the cash price of a commodity is greater
// than a futures price, the market is backward or in backwardation.
// convenience yield - The benefit of being able to obtain a commodity right now. eg If the cash price in the marketplace is actually $76.25 and the cash price of the contract is 75$,
// the convenience yield ought to be $1.25. This is the additional amount users are willing to pay for the benefit of having immediate access to the commodity.
// arbitrage - the buying and selling of the same or very closely related instruments in different markets to profit from an apparent mispricing.
// implied spot price - If we know the forward price, time to maturity, interest rate, and dividend, we can solve for S, the implied spot price of the underlying asset
// ** same applies for implied interest rate and implied dividend
// Dividends - Declared Date - the date on which a company announces both the amount of the dividend and the date on which the dividend will be paid. Once the company declares the dividend. the dividend risk is
// eliminated, at least until the next dividend payment.
// Dividends - Record Date - the date on which the stock must be owned in order to receive the dividend. Regardless of the date on which the stock is purchased, ownership of the stock
// does not become official until the settlement date, the date on which the purchaser of the stock officially takes possession. In the USA, the settlement date for stock is normally
// three business days after the trade is made.
// Dividends - Ex-Dividend Date (Ex-Date) - The first day on which a stock is trading without the rights to the dividend. In the United states, the last day on which a stock can be
// purchased in order to receive the dividend is three business days prior to the record date. The ex-dividend date is two business days prior to the record date.
// Dividends - Payable Date - The date on which the dividend will be paid to qualifying shareholders(those owning shares on the record date).
// locked market - Some futures exchanges have daily price limits for futures contracts. When a futures contract reaches this limit, the market is said to be locked or locked limit
// If the market is either limit up or limit down, no further trading may take place until the price comes off the limit (someone is willing to sell at a price equal to or less
// than the up limit or buy at a price equal to or higher than the down limit)
// sell stock short - sell stock that he does not already own.
// ** a trader who wants to sell stock short must first borrow the stock.
// short-stock rebate - The rate that the trader receives on the short sale of stock.
// ** Depending on how difficult is it to borrow a certain stock the interest rate varies.
// Contract Specifications - Type - A call option is the right to buy or take a long position in an asset at a fixed price on or before a specified date. A put option is the right to sell
// or take a short position in an asset.
// Contract Specifications - Underlying - the underlying asset or, more simply, the underlying is the security or commodity to be bought or sold under the terms of the option contract.
// ** One hundred shares is sometimes referred to as a round lot. An order to buy or sell fewer than 100 shares is an odd lot.
// ** Many exchanges also permit trading flex options, where the buyer and seller may negotiate the contract specifications, including the quantity of the underlying, the expiration date,
// the exercise price, and the exercise style.
// ** Most often, the underlying for an option on a futures contract is the futures month that corresponds to the expiration month of the option.
// serial options on futures - option expirations where there is no corresponding futures month.
// midcurve options - short-term options on long-term futures. eg one-year midcurve
// Contract Specifications - Expiration Date or Expiry - The expiration date is the date on which the owner of an option must make the final decision whether to buy, in the case of a call, or to sell, in
// the case of a put.
// ** On many stock option exchanges, the expiration date for stock and stock index options is the third friday of the expiration month.
// last trading date - the last business date prior to expiration on which an option can be bought or sold on an exchange.
