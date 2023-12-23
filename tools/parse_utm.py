import utm

class UWConvert:
    def __init__(self, offset: str = "data/offset.txt"):
        self.offset_file = offset
        self.utm_zone = None
        self.hemisphere = None
        self.x_coordinate = None
        self.y_coordinate = None

        self.get_utm(self.offset_file)

    def get_utm(self, path):
        with open(path, 'r') as file:
            content = file.read()
        values = content.split()
        self.utm_zone = int(values[2][:2])
        self.hemisphere = values[2][-1]
        self.x_coordinate = float(values[-2])
        self.y_coordinate = float(values[-1])

    def W2U(self, wgs84: list):  # [lat,lon,alt]  to [x,y,z]
        utm_point = utm.from_latlon(
            wgs84[0], wgs84[1], self.utm_zone, self.hemisphere)
        return [utm_point[0] - self.x_coordinate, utm_point[1]-self.y_coordinate, wgs84[2]]

    def U2W(self, utm_o: list):  # [x,y,z]  to [lat,lon,alt]
        utm_point = utm.to_latlon(
            utm_o[0]+self.x_coordinate, utm_o[1]+self.y_coordinate, self.utm_zone, self.hemisphere)
        return [utm_point[0], utm_point[1], utm_o[2]]


if __name__ == "__main__":
    c = UWConvert("data/JiuLongLake_1223/JiuLongLake_1223_offset.txt")

    point1 = [23.468897, 113.400119, 56.835567]
    point2 = [23.468966, 113.399027, 56.807884]
    point3 = [23.470517, 113.400007, 45.802860]
    point4 = [23.469606, 113.399397, 74.113525]
    point5 = [23.468785, 113.401240, 62.445034]

    points = [point1, point2, point3, point4, point5]
    for i in points:
        print([*c.W2U(i[:2]), i[2]])
