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

    def W2U(self, wgs84: list):
        utm_point = utm.from_latlon(*wgs84, self.utm_zone, self.hemisphere)
        return [utm_point[0] - self.x_coordinate, utm_point[1]-self.y_coordinate]

    def U2W(self, utm_o: list):
        utm_point = utm.to_latlon(
            utm_o[0]+self.x_coordinate, utm_o[1]+self.y_coordinate, self.utm_zone, self.hemisphere)
        return utm_point


if __name__ == "__main__":
    c = UWConvert("data/offset.txt")
    print(c.W2U([31.263, 121.638]))
