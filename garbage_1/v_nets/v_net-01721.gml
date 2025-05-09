graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 1721
  arrival_time 38465.64268052724
  lifetime 1434.0444280568795
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 15
    gpu 0
    rom 5
  ]
  node [
    id 1
    label "1"
    cpu 14
    gpu 46
    rom 6
  ]
  node [
    id 2
    label "2"
    cpu 30
    gpu 18
    rom 1
  ]
  node [
    id 3
    label "3"
    cpu 43
    gpu 40
    rom 41
  ]
  node [
    id 4
    label "4"
    cpu 35
    gpu 49
    rom 11
  ]
  node [
    id 5
    label "5"
    cpu 29
    gpu 33
    rom 25
  ]
  node [
    id 6
    label "6"
    cpu 41
    gpu 44
    rom 0
  ]
  node [
    id 7
    label "7"
    cpu 33
    gpu 31
    rom 35
  ]
  node [
    id 8
    label "8"
    cpu 48
    gpu 37
    rom 4
  ]
  node [
    id 9
    label "9"
    cpu 18
    gpu 18
    rom 6
  ]
  node [
    id 10
    label "10"
    cpu 18
    gpu 47
    rom 17
  ]
  node [
    id 11
    label "11"
    cpu 40
    gpu 18
    rom 47
  ]
  node [
    id 12
    label "12"
    cpu 21
    gpu 15
    rom 8
  ]
  node [
    id 13
    label "13"
    cpu 35
    gpu 34
    rom 30
  ]
  node [
    id 14
    label "14"
    cpu 47
    gpu 9
    rom 17
  ]
  edge [
    source 0
    target 1
    bw 42
  ]
  edge [
    source 1
    target 2
    bw 39
  ]
  edge [
    source 2
    target 3
    bw 26
  ]
  edge [
    source 3
    target 4
    bw 32
  ]
  edge [
    source 4
    target 5
    bw 45
  ]
  edge [
    source 5
    target 6
    bw 31
  ]
  edge [
    source 6
    target 7
    bw 35
  ]
  edge [
    source 7
    target 8
    bw 34
  ]
  edge [
    source 8
    target 9
    bw 45
  ]
  edge [
    source 9
    target 10
    bw 4
  ]
  edge [
    source 10
    target 11
    bw 1
  ]
  edge [
    source 11
    target 12
    bw 29
  ]
  edge [
    source 12
    target 13
    bw 24
  ]
  edge [
    source 13
    target 14
    bw 46
  ]
]
