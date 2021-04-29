import pandas as pd
from datetime import datetime
import numpy as np
from typing import List
import matplotlib.pyplot as plt


def datetime_parser(datestring):
    return datetime.strptime(datestring, '%Y-%m-%d')


def get_preprocessed_historical_orders(company: str = None) -> pd.DataFrame:
    df_orders = pd.read_json('../../data/orders/deliveries.json', encoding='utf-8')

    # Set company
    if company:
        assert company in ["Mowi Feed AS", "BioMar AS"], f"Unknown company {company}"
        df_orders = df_orders[df_orders["division name"] == company]

    # Drop columns
    df_orders = df_orders.drop(columns=['contactNo1', 'contactNo2', 'contactNo3', 'product attribute',
                                        'pickup warehouse id'])

    # Parse columns
    df_orders = df_orders.dropna(subset=['departure date'])  # TODO: Any useful rows dropped here?
    df_orders['departure date'] = df_orders['departure date'].apply(datetime_parser)
    df_orders['supplierNo'] = df_orders['supplierNo'].astype('Int64').astype('str')
    df_orders['shipment'] = df_orders['shipment'].astype('Int64').astype('str')
    df_orders['mmsi id'] = df_orders['mmsi id'].astype('Int64').astype('str')

    # Filter to order rows
    df_orders = df_orders[df_orders['supplierNo'].astype('int64') > 2100]  # skip factory visits
    df_orders = df_orders[df_orders['qty kg'] > 100]  # cut away some tiny orders
    df_orders = df_orders[df_orders['product name'] != "Empty bags"]  # remove some Mowi orders for empty bags

    return df_orders


def get_random_products(n: int, products: List[str], distr: str = 'triangular') -> List[str]:
    if distr == 'triangular':
        return [products[int(np.random.triangular(0, 0, len(products)))] for _ in range(n)]
    elif distr == 'uniform':
        return np.random.choice(products, n)


def sample_orders_df(n: int = None, company: str = None, no_products: int = 10) -> pd.DataFrame:
    orders = get_preprocessed_historical_orders(company=company)

    orders['shipment_supplierNo'] = orders['shipment'] + '_' + orders['supplierNo']
    orders['no products'] = np.ones(len(orders), dtype=int)
    grouped_orders = orders.groupby(['shipment_supplierNo', 'shipment', 'supplierNo']).sum()
    if n == -1:
        n = len(grouped_orders)
    order_sample = grouped_orders.sample(n=n).reset_index()

    sample_shipment_supplierNo = order_sample['shipment_supplierNo'].values

    products = [f'p_{i}' for i in range(1, no_products + 1)]
    data = {'supplierNo': []}
    data.update({product: [] for product in products})
    for shipment_supplier in sample_shipment_supplierNo:
        shipment, supplier = shipment_supplier.split("_")
        data['supplierNo'].append(supplier)
        for product in products:
            data[product].append(0)

        relevant_orders = orders[orders['shipment_supplierNo'] == shipment_supplier]
        random_products = get_random_products(len(relevant_orders), products)
        quantities = relevant_orders['qty kg'].values
        for product, qty in zip(random_products, quantities):
            data[product][-1] += qty // 1000
    df = pd.DataFrame(data).set_index('supplierNo')

    # print(df)
    # sum_p = df[products].sum().sort_values(ascending=False)
    # print(sum_p)
    # plt.plot(sum_p)
    # plt.show()

    return df


def _describe_grouped_orders(orders: pd.DataFrame) -> None:
    print(orders.head(20))
    print()
    print(orders.tail(20))
    plt.hist(orders['qty kg'], bins=10)
    plt.show()
    plt.hist(orders['no products'], bins=int(orders['no products'].max()))
    plt.show()
    print(orders.describe())


if __name__ == '__main__':
    df = sample_orders_df(n=32, company=None)
    file_path = '../../data/input_data/testcase.xlsx'
    excel_writer = pd.ExcelWriter('../../data/test.xlsx', engine='openpyxl', mode='w')
    # with excel_writer as writer:
    #     df.to_excel(writer, sheet_name='test')
    df.to_excel(excel_writer, sheet_name='test')
    excel_writer.save()

    # Save locations
    # locations_df = pd.DataFrame({'location id': list(df2['supplierNo'].unique())})
    # file_path = '../data/locations.xlsx'
    # excel_writer = pd.ExcelWriter(file_path, engine='openpyxl')
    # excel_writer.book = openpyxl.load_workbook(file_path)
    # locations_df.to_excel(excel_writer, sheet_name='Ark1')
    # excel_writer.save()
    # excel_writer.close()
